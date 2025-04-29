from flask import Flask, request, render_template, redirect
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


# Initialize Flask app, serving static files from 'static' folder
app = Flask(__name__, static_folder='static')


def compute_SI(age: float, sex: str, weight: float) -> tuple:
    """
    Compute insulin sensitivity (SI) and total daily dose (TDD).

    SI (mL/µU/min) is scaled by a reference sensitivity and user TDD.
    TDD (IU/day) is estimated based on age, sex, and weight per literature estimations (Lebbad et al.).
    """
    # Total daily dose calculation differs by sex
    if sex.lower() == 'female':
        TDD = 16.87 - 0.59 * age + 0.53 * weight + 0.42 * age * weight / 100
    else:
        TDD = (
            16.87
            - 0.59 * age
            + 0.53 * weight
            + 0.42 * age * weight / 100
            - 10.18
            + 0.19 * weight
        )

    # Reference sensitivity constant from Bergman et al., 1981
    SI_REF = 4.5e-4       # mL/µU/min

    # Estimated weight-based reference TDD
    TDD_REF = weight * 0.55  # IU/day; multiplicative factor adapted from https://www.wcu.edu/WebFiles/PDFs/CalculatingInsulin.pdf

    # Scale sensitivity by ratio of reference to user TDD ; based on inverse proportionality of the two parameters as per the 1800 Rule
    SI = (SI_REF * TDD_REF) / TDD
    return SI, TDD


def u_glucose(t, carb_g, meal_time, tau= 30.0): #tau chosen as per literature references: https://www.academia.edu/21213820/Insulin_Therapy_and_Hypoglycemia
    """
    Exponential glucose appearance rate (u2) in mg/min.
    - No input before meal_time.
    - After meal_time, glucose appears exponentially with time constant tau.
    """
    if t < meal_time:
        return 0.0

    carb_mg = carb_g * 1000.0  # Convert grams to mg
    return (carb_mg / tau) * np.exp(-(t - meal_time) / tau)


def minimal_model_full(t, y, SI, weight, carb_amount, meal_time, ka=0.02, ke=0.012):
    """
    Four-state minimal model of glucose-insulin kinetics.

    States:
      G: Plasma glucose concentration(mg/dL)
      X: Remote insulin action
      I: Plasma insulin concentration(µU/mL)
      S: Insulin subcutaneous depot (IU)

    Equations:
      dG/dt = -p1*(G - G_b) - X*G + u2/Vol_g
      dX/dt = -p2*X + p3*(I - I_b)
      dI/dt = ka*S*1000/V_I - ke*(I - I_b)
      dS/dt = -ka*S
    """
    G, X, I, S = y

    # Model parameters
    p1, p2 = 0.035, 0.05          # Rate constants (1/min)
    p3 = SI * p2                  # Insulin action parameter (mL/µU/min^2)
    G_b, I_b = 80.0, 12.2         # Basal glucose and insulin
    Vol_g = 117.0                 # Glucose distribution volume (dL)

    # Input rates
    u2 = u_glucose(t, carb_amount, meal_time)
    V_I = (0.15 * weight) * 10    # Insulin distribution volume (dL) adjusted per BW

    # Differential equations
    dG_dt = -p1 * (G - G_b) - X * G + u2 / Vol_g
    dX_dt = -p2 * X + p3 * (I - I_b)
    dI_dt = ka * S * 1000 / V_I - ke * (I - I_b)
    dS_dt = -ka * S

    return [dG_dt, dX_dt, dI_dt, dS_dt]



def plot_time_series(t, y, xlabel, ylabel, title, filename, secondary_y=None):
    """
    Plot a basic time series and save to file.

    If 'secondary_y' is provided, overlay a second data series on a twin y-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, linewidth=2)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



@app.route('/')
def home():
    """Redirect root URL to simulator page."""
    return redirect('/simulator')


@app.route('/simulator', methods=['GET', 'POST'])
def simulator():
    """
    Handle simulator form input, run model, generate plots, and render results.
    """
    # Context dictionary for template rendering
    ctx = {
        'glucose_plot': None,
        'insulin_plot': None,
        'overlay_plot': None,
        'SI': None,
        'TDD': None,
        'error': None,
    }

    if request.method == 'POST':
        try:
            # Parse inputs from form
            age             = float(request.form['age'])
            sex             = request.form['sex']
            weight          = float(request.form['weight'])
            bolus_insulin   = float(request.form['bolus_insulin'])
            injection_time  = float(request.form['injection_time'])
            carb_amount     = float(request.form['carb_amount'])
            meal_time       = float(request.form['meal_time'])
        except (ValueError, KeyError):
            ctx['error'] = 'Please enter valid numbers in every field.'
            return render_template('simulator.html', **ctx)

        # Compute patient-specific parameters
        SI, TDD = compute_SI(age, sex, weight)
        ctx['SI']  = f"{SI:.4e} mL/µU/min"
        ctx['TDD'] = f"{TDD:.1f} IU/day"

        # Solve ODE in two phases: before and after insulin bolus
        t_final = 360  # Total simulation duration (min)
        # High resolution around injection
        t_eval_pre = np.linspace(0, injection_time, 500)
        t_eval_post = np.linspace(injection_time, t_final, 1000)

        # Initial state at t=0
        y0 = [80.0, 0.0, 12.2, 0.0]

        # Phase 1: pre-injection
        sol_pre = solve_ivp(
            minimal_model_full,
            [0, injection_time],
            y0,
            args=(SI, weight, carb_amount, meal_time),
            t_eval=t_eval_pre
        )

        # Add bolus insulin to subcutaneous depot
        G_inj, X_inj, I_inj, S_inj = sol_pre.y[:, -1]
        S_inj += bolus_insulin
        y_inj = [G_inj, X_inj, I_inj, S_inj]

        # Phase 2: post-injection
        sol_post = solve_ivp(
            minimal_model_full,
            [injection_time, t_final],
            y_inj,
            args=(SI, weight, carb_amount, meal_time),
            t_eval=t_eval_post
        )

        # Combine results
        t_full = np.concatenate([sol_pre.t, sol_post.t])
        y_full = np.concatenate([sol_pre.y, sol_post.y], axis=1)

        # File names with timestamp to avoid collisions
        ts = str(int(time.time()))
        files = {
            'glucose_plot':  f'static/glucose_{ts}.png',
            'insulin_plot':  f'static/insulin_{ts}.png',
            'overlay_plot':  f'static/overlay_{ts}.png',
        }

        # Generate and save plots
        # Glucose
        plot_time_series(
            t_full, y_full[0],
            xlabel='Time (min)',
            ylabel='Glucose (mg/dL)',
            title='Glucose Dynamics',
            filename=files['glucose_plot']
        )
        # Insulin
        plot_time_series(
            t_full, y_full[2],
            xlabel='Time (min)',
            ylabel='Insulin (µU/mL)',
            title='Insulin Dynamics',
            filename=files['insulin_plot']
        )

        # Overlay with dual y-axis
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t_full, y_full[0], 'b-', linewidth=2)
        ax1.set_xlabel('Time (min)', fontsize=14)
        ax1.set_ylabel('Glucose (mg/dL)', color='b', fontsize=14)
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_ylim(50, 300)
        ax2 = ax1.twinx()
        ax2.plot(t_full, y_full[2], 'r-', linewidth=2)
        ax2.set_ylabel('Insulin (µU/mL)', color='r', fontsize=14)
        plt.title('Glucose & Insulin Overlay', fontsize=16, fontweight='bold')
        fig.tight_layout()
        plt.savefig(files['overlay_plot'], bbox_inches='tight')
        plt.close(fig)

        # Pass filenames to template
        for key, path in files.items():
            ctx[key] = os.path.basename(path)

    return render_template('simulator.html', **ctx)


@app.route('/fasting-instructions')
def fasting_instructions():
    """Render fasting instructions page."""
    return render_template('fasting_instructions.html')


@app.route('/about')
def about():
    """Render about page."""
    return render_template('about.html')


@app.route('/disclaimer')
def disclaimer():
    """Render disclaimer page."""
    return render_template('disclaimer.html')


if __name__ == '__main__':
    # Run Flask app in debug mode for development
    app.run(debug=True)

from include import *
# DISCLAIMER: This file is a collection of the functions used mostly in the AppStat course.

def create_pdf_err(data, n_bins, bin_range):
    """Creates the bin_centers, density and the poisson errors for a density=True histogram. This allows to create errorbars on a pdf"""
    # Poisson errors (These occur since all measurements are independent
    # so the expected value in a specific bin is given by its small probability of being in that exact bin)
    density, bins = np.histogram(data, bins=n_bins, density=True, range=bin_range)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bw = bins[1] - bins[0]
    pdf_err = np.sqrt(density/(len(data) * bw))      # Since the distribution is a pdf, 
                                                     # the poisson errors also needs to be rescaled 
                                                     # slightly: \sigma_p = \sqrt{p/(N \Delta x)}
                                                                        
    return bin_centers, density, pdf_err # This allows to create errorbars on a pdf


# def simple_err_prop(derivatives, uncertainties):
#     """Should use np.array"""
#     # Convert the derivatives from sympy.Float to reg float()
#     reg_derivatives = []
#     for val in derivatives:
#         val = float(val)            
#         reg_derivatives.append(val)
#     comb_Var = np.array(reg_derivatives)**2 * np.array(uncertainties)**2
#     err = np.sqrt(np.sum(comb_Var, axis=0))
    
#     return err
def Multivar_err_prop(deriv_vals, uncertainties):
    """Computes the error propagation for a multivariate function, given the derivative values and the uncertainties of the variables.
    ARGS:
        deriv_vals: a 2D array containing the derivative values for each variable and each data point (shape: n_variables x n_data_points)
        uncertainties: a 2D array containing the uncertainties for each variable and each data point (shape: n_variables x n_data_points)
        """
    print("IMPORTANT: THIS ASSUMES NO CORRELATION BETWEEN VARIABLES.\nONLY USE FOR INDEPENDENT VARIABLES.")
    # Ensure the shapes are compatible
    if deriv_vals.shape != uncertainties.shape:
        raise ValueError(f"Shape of derivative values {deriv_vals.shape} does not match shape of uncertainties {uncertainties.shape}")
    
    # Compute the error propagation using the formula: sigma_f = sqrt( sum( (df/dx_i * sigma_x_i)^2 ) )
    squared_terms = (deriv_vals * uncertainties) ** 2
    print(np.shape(squared_terms))
    total_variance = np.sum(squared_terms, axis=0)      # Sums up the contributions from all variables for each data point
    
    return np.sqrt(total_variance)

def weighted_avg(values, errors):
    values = np.array(values)
    errors = np.array(errors)
    mean = (np.sum((values/errors**2))/(np.sum(1/errors**2))) # From eq. 4.6 Barlow
    Var = (1/(np.sum(1/errors**2)))                           # From eq. 4.7 Barlow
    err = np.sqrt(Var)
    
    return mean, err


def get_chi2(data, model_fit, errors, n_params):
    """Calculates the chi2 value and the corresponding p-value
    for a given dataset, fitted model, errors and number of parameters."""
    chi2 = np.sum((data - model_fit)**2 / errors**2)
    dof = len(data) - n_params
    prob = stats.chi2.sf(chi2, dof)
    return chi2, prob


def get_derivatives(expression, parameters, values, evaluate=True):
    """Using an expression and parameters to be differentiated.
       Should obtain the derivatives in some way that can be translated into values 
       Arguments:
       expression: sympy expression (i.e. a*x + b)
       parameters: list of sympy symbols (i.e. a = sp.symbols('a'))
       values: tuple of values for the parameters (i.e. (1, 2))"""
    # Get the derivatives for all the different parameters
    deriv_expressions = []
    for i in range(len(parameters)):
        deriv_expressions.append(expression.diff(parameters[i]))
        
    # Convert the derivatives to numerical functions that can be called
    deriv_funcs = [sp.lambdify(parameters, derivative) for derivative in deriv_expressions]
    
    if evaluate:
        # Evaluate the derivatives at the example values
        evaluated_derivatives = [func(*values) for func in deriv_funcs]
    else:
        evaluated_derivatives = None
        
    return np.array(deriv_expressions), np.array(evaluated_derivatives), deriv_funcs



def get_fit_text(mfit, parameters, points, get_chi2=True, Large_num = False):
    """Obtaining the values for the fit parameters and the chi2 value and turning them into a string
       Args: 
         mfit: Minuit object
         parameters: List of strings with the names of the parameters
         points: The data points that were fitted
    """
    
    fit_vals = []
    
    if Large_num:
        for i in range(len(mfit.values)):
            fit_vals.append(f"{parameters[i]}  :  {mfit.values[i]:.3e} ± {mfit.errors[i]:.3e}")
    else:
        for i in range(len(mfit.values)):
            fit_vals.append(f"{parameters[i]}  :  {mfit.values[i]:.3f} ± {mfit.errors[i]:.3f}")

    if get_chi2:
        chi2 = mfit.fval
        ndf = len(points) - len(mfit.values)
        prob = stats.chi2.sf(chi2, ndf)
        fit_text = "Fit Results: \n" + "\n".join(fit_vals) + f"\n$\\chi^2$  :  {chi2:.3f}" + "\n" +  r"$N_{dof}$" + f"  :  {ndf:.3f}" +  "\n" + r"$P(\chi^2 ; N_{dof})$ :  " + f"{prob:.3f}"
    else:
        fit_text = "Fit Results: \n" + "\n".join(fit_vals)
    return fit_text


def put_fit_text(fit_text, ax, ax_loc, text_params, ha = "left"):
    """Putting the fit text into the plot
       Args:
         fit_text: The text to be put into the plot
         ax: The axis object
         ax_loc: The location of the text in the plot
         
       Example:
        ax_loc = (1,2.5)
        text_params = {"fontsize": 10, "color": "b"}
        put_fit_text(fit_text, plt, ax_loc, text_params)
    """
    # Setting the text box    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in axes coords
    ax.text(ax_loc[0], ax_loc[1], fit_text, text_params, bbox = props, ha = ha)



   
def plot_grouped_data(data_list, colors, x_axes, y_axis, figsize=(12, 8), title=None, axtitles=None, n_stddevs = 1):
    """Create a function for grouping data together nicely in viewable plots. Assume a (nx2) structure.
       DISCLAIMER: Only here to show an example of how we could present data."""
    
    # Constructing the number of rows and columns for the subplots
    if len(data_list) == 1:
        n_rows, n_cols = 1, 1
    else:
        n_rows = int(len(data_list)/2) + len(data_list)%2
        n_cols = 2
    
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    ax = ax.flatten()  # Flatten the array of axes for easy iteration
    
    for i, data in enumerate(data_list):
        # Data analysis
        x, y = data
        mean_y = np.mean(y)
        std = np.std(y)
        
        
        # Actual plotting
    
        ax[i].plot(x, y, color=colors[i], alpha=0.25, label='Data')
        
        ax[i].axhline(mean_y, color='k', linestyle='--', label=f'Mean: {mean_y:.2f}')  
        ax[i].axhspan(mean_y - 2*std, mean_y + 2*std, color=colors[i], alpha=0.4, label=f'95%-Confidence Interval')
        
        ax[i].set_xlabel(x_axes[i])
        ax[i].set_ylabel(y_axis[i])
        ax[i].legend(loc = "upper right", fontsize=10)
        
        ax[i].set_ylim(np.min(y) - n_stddevs*np.std(y), np.max(y) + n_stddevs*np.std(y))

        
    # Extra layout
    if title is not None:
        plt.suptitle(title)  
    
    for i in range(len(data_list)):
        if axtitles is not None:        
            ax[i].set_title(axtitles[i])
    
    plt.tight_layout()
    return fig, ax

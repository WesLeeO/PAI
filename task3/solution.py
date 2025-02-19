"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
from scipy.stats import norm
import math

# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # We initialize the 2 Gaussian Processes with kernels prposed in the task description
        # We also added a White Kernel to both GPs to account for the noisy obeservations.
        self.f = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.15**2))
        self.v = GaussianProcessRegressor(kernel= DotProduct(sigma_0=0)+ Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.0001**2))
        self.max_sa = SAFETY_THRESHOLD
        # We experimentally optimized the hyperparameters for the weight of the std in the acquisition function and the Lipchitz constant
        self.L = 12
        self.beta_ucb = 1.0
        self.inputs = []
        self.f_values = []
        self.v_values = []
        self.safe_set = set()

        # We discretized the interval of reals from 0 to 10 by choosing 1000 points equally spaced inn this interval
        self.D = np.linspace(0, 10, 1000)
    

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # We return the point from our safe set that maximises the acquistion function (GP-UCB)
        # We do safe optimization
        maximum = float('-inf')
        point = None
        for safe_point in self.safe_set:
            mean, std = self.f.predict(np.atleast_2d(safe_point), return_std=True)
            acquisition = mean + self.beta_ucb * std
            if acquisition > maximum:
                maximum = acquisition
                point = safe_point
        return np.array(point)


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            #x0 is starting point
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            #* unpacks the tuple anc clip makes it belong to the interval
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        # scalar from (1,) narray
        x_opt = x_values[ind].item()
        return x_opt

    def acquisition_function(self, x: np.ndarray):
        #GP-UCB
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        # TODO: Implement the acquisition function you want to optimize.

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """

        self.safe_set.add(x.item())
        self.inputs.append(x.item())
        self.f_values.append(f.item())
        # This tranformation allows to forget about the prior on the mean
        # and have safe samples a value > 0 (helpful for the constraint when we update the safe set)
        self.v_values.append(self.max_sa - v.item())
        formatted_inputs = np.array(self.inputs).reshape(-1, 1)
        self.f.fit(formatted_inputs, np.array(self.f_values))
        self.v.fit(formatted_inputs, np.array(self.v_values))
        self.update_safe_set()
    

    def update_safe_set(self):
        #Extend our safe set by choosing points in the neighborhood of the safe ones (depends on self.L)
        new_safe_set = self.safe_set.copy()
        for safe_point in self.safe_set:
            mean, std = self.v.predict(np.atleast_2d(safe_point), return_std=True)
            lower_bound = mean - std
            for point in self.D:
                if abs(point - safe_point) <= lower_bound / self.L:
                    new_safe_set.add(point)
        self.safe_set = new_safe_set
    
        


    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        assert(len(self.inputs) == len(self.f_values))
        assert(len(self.f_values) == len(self.v_values))
        safe_inputs = [x for i, x in enumerate(self.inputs) if self.v_values[i] >= 0]
        safe_values = [f for f, v in zip(self.f_values, self.v_values) if v >= 0]
        max_index = np.argmax(safe_values)
        #print(f"Returned: {safe_inputs[max_index]}")
        return safe_inputs[max_index]

        #raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

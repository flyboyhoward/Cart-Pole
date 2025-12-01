import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from scipy.signal import place_poles
import matplotlib.pyplot as plt


class CartPoleSwingUpEnv(CartPoleEnv):
    """
    Swing-up variant of Gymnasium's Classic Control CartPoleEnv.

    - Continuous action: horizontal force on the cart (N)
    - Starts pole near downward configuration (theta ≈ pi)
    - Can use Gymnasium's built-in visualization (render_mode="human")
    """

    def __init__(self, render_mode="human"):
        super().__init__(render_mode=render_mode)

        # Continuous action in [-max_force, max_force]
        self.max_force = 10.0
        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float32,
        )

        # Allow large motion; we'll wrap theta manually
        self.theta_threshold_radians = math.pi
        self.x_threshold = 5.0

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)

        # Start near the downward position (theta ≈ pi)
        theta = math.pi + self.np_random.uniform(-0.05, 0.05)
        self.state = np.array(
            [
                0.0,   # x
                0.0,   # x_dot
                theta, # theta (downward)
                0.0,   # theta_dot
            ],
            dtype=np.float64,
        )

        return np.array(self.state, dtype=np.float32), info

    def step(self, action):
        # Continuous force
        force = float(np.clip(action[0], -self.max_force, self.max_force))

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Classic CartPole dynamics, but with continuous force
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # Wrap angle to [-pi, pi] so it doesn't drift
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        # No termination for this control demo; we just run for a fixed time
        terminated = False
        truncated = False
        reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    # Continuous-time dynamics x_dot = f(x, u) for linearisation
    def continuous_dynamics(self, state, u):
        x, x_dot, theta, theta_dot = state
        force = float(u)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        return np.array([x_dot, xacc, theta_dot, thetaacc], dtype=np.float64)


# ---------- Linearisation & pole-placement ----------

def linearize_cartpole(env: CartPoleSwingUpEnv):
    """
    Linearise the continuous-time cart-pole dynamics around the upright equilibrium.
    Returns A, B for x_dot = A x + B u.
    """
    n = 4
    m = 1

    x_eq = np.zeros(n)  # [x, x_dot, theta, theta_dot] = [0, 0, 0, 0] upright, centered
    u_eq = 0.0

    def f(x, u):
        return env.continuous_dynamics(x, u)

    eps = 1e-5
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # Numerical partials for A
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (f(x_eq + dx, u_eq) - f(x_eq - dx, u_eq)) / (2 * eps)

    # Numerical partials for B
    du = 1e-5
    B[:, 0] = (f(x_eq, u_eq + du) - f(x_eq, u_eq - du)) / (2 * du)

    return A, B


def design_pole_placement(A, B):
    """
    Classic control design:
        Place closed-loop poles at chosen locations.
    """
    desired_poles = [-2.0, -2.5, -3.0, -3.5]
    pp = place_poles(A, B, desired_poles)
    K = pp.gain_matrix
    return K


# ---------- Energy-based swing-up ----------

def energy_based_swing_up(env: CartPoleSwingUpEnv, state):
    """
    Energy-based swing-up controller.

    - Define energy with respect to the *downward* configuration.
    - Desired energy corresponds to the *upright* configuration.
    """
    x, x_dot, theta, theta_dot = state
    theta_tilde = theta  # theta = 0 is upright (from env.step wrapping)

    # Represent angle with downward as zero
    theta_down = theta_tilde + math.pi

    m = env.masspole
    g = env.gravity
    l = env.length * 2.0  # full pole length

    # Mechanical energy with downward config as reference:
    E = 0.5 * m * (l ** 2) * theta_dot ** 2 + m * g * l * (1 - math.cos(theta_down))

    # Desired energy = energy of upright (theta_down = pi)
    E_desired = m * g * l * (1 - math.cos(math.pi))  # = 2 m g l

    # Gains
    k_e = 0.5
    k_x = 0.2
    k_xdot = 0.2

    prod = theta_dot * math.cos(theta_tilde)
    if prod == 0:
        sign_term = 0.0
    else:
        sign_term = math.copysign(1.0, prod)

    # Energy-shaping term
    u = k_e * (E - E_desired) * sign_term
    # Keep cart near origin
    u += -k_x * x - k_xdot * x_dot

    return u


# ---------- State-feedback hybrid (pole placement) ----------

def hybrid_state_feedback_controller(env: CartPoleSwingUpEnv, state, K, use_stabiliser):
    """
    Hybrid controller for state-feedback (pole placement):

    - Far from upright: energy-based swing up.
    - Near upright: state-feedback stabiliser with given K.
    """
    x, x_dot, theta, theta_dot = state
    theta_tilde = theta  # angle from upright in [-pi, pi]

    theta_switch = 0.35      # ~20 deg
    theta_dot_switch = 1.5   # rad/s

    if use_stabiliser:
        # In stabilisation mode: u = -K x
        x_vec = np.array([x, x_dot, theta_tilde, theta_dot])
        u = - (K @ x_vec)[0]
        mode = "stabiliser"
    else:
        # Check if we're close enough to switch to stabiliser
        if abs(theta_tilde) < theta_switch and abs(theta_dot) < theta_dot_switch:
            use_stabiliser = True
            x_vec = np.array([x, x_dot, theta_tilde, theta_dot])
            u = - (K @ x_vec)[0]
            mode = "stabiliser"
        else:
            # Energy-based swing-up
            u = energy_based_swing_up(env, state)
            mode = "energy"

    # Saturate
    u = float(np.clip(u, -env.max_force, env.max_force))
    return u, use_stabiliser, mode


# ---------- PID-style stabiliser ----------

def pid_stabiliser(env: CartPoleSwingUpEnv, state, pid_state, gains, dt):
    """
    PID-style stabiliser for upright regulation.

    - PD on pole angle (theta, theta_dot)
    - PI(D) on cart position (x, x_dot) with integral on x
    """
    x, x_dot, theta, theta_dot = state

    # References
    x_ref = 0.0
    theta_ref = 0.0

    # Errors
    e_theta = theta_ref - theta          # want theta -> 0
    e_theta_dot = 0.0 - theta_dot
    e_x = x_ref - x                      # want x -> 0
    e_x_dot = 0.0 - x_dot

    # Update integrator on x error
    pid_state["x_int"] += e_x * dt

    # Gains
    Kp_theta = gains["Kp_theta"]
    Kd_theta = gains["Kd_theta"]
    Ki_theta = gains["Ki_theta"]

    Kp_x = gains["Kp_x"]
    Kd_x = gains["Kd_x"]
    Ki_x = gains["Ki_x"]

    # Angle PD (+ optional I for theta if needed; off here for stability)
    u_theta = Kp_theta * e_theta + Kd_theta * e_theta_dot + Ki_theta * 0.0

    # Cart PI(D)
    u_x = Kp_x * e_x + Kd_x * e_x_dot + Ki_x * pid_state["x_int"]

    u = u_theta + u_x

    # Saturate
    u = float(np.clip(u, -env.max_force, env.max_force))
    return u, pid_state


def hybrid_pid_controller(env: CartPoleSwingUpEnv, state, pid_state, gains, use_pid):
    """
    Hybrid controller for PID:

    - Far from upright: energy-based swing up.
    - Near upright: PID-style stabiliser.
    """
    x, x_dot, theta, theta_dot = state
    theta_tilde = theta

    theta_switch = 0.35
    theta_dot_switch = 1.5

    dt = env.tau

    if use_pid:
        u, pid_state = pid_stabiliser(env, state, pid_state, gains, dt)
        mode = "PID"
    else:
        if abs(theta_tilde) < theta_switch and abs(theta_dot) < theta_dot_switch:
            use_pid = True
            u, pid_state = pid_stabiliser(env, state, pid_state, gains, dt)
            mode = "PID"
        else:
            u = energy_based_swing_up(env, state)
            mode = "energy"

    return u, pid_state, use_pid, mode


# ---------- EPISODE RUNNERS WITH LOGGING (PID & Pole placement) ----------

def run_pid_episode(env, label="PID controller", total_time=16.0, render=True):
    dt = env.tau
    num_steps = int(total_time / dt)

    obs, _ = env.reset()
    use_pid = False

    print(f"\n=== Running episode with controller: {label} ===")

    # PID gains (tune as desired)
    pid_gains = {
        "Kp_theta": 80.0,
        "Kd_theta": 12.0,
        "Ki_theta": 0.0,  # keep theta integral off for stability
        "Kp_x": 1.0,
        "Kd_x": 0.5,
        "Ki_x": 0.1,      # small integral on x to recenter cart
    }

    pid_state = {"x_int": 0.0}

    # Track "stable upright" time
    stable_counter = 0
    stable_required_steps = int(2.0 / dt)  # 2 seconds

    # Logs
    t_log, x_log, theta_log, u_log = [], [], [], []

    for k in range(num_steps):
        t = k * dt
        u, pid_state, use_pid, mode = hybrid_pid_controller(
            env, obs, pid_state, pid_gains, use_pid
        )

        x, x_dot, theta, theta_dot = obs

        # Logging
        t_log.append(t)
        x_log.append(x)
        theta_log.append(theta)
        u_log.append(u)

        # Stability check
        if use_pid and abs(theta) < 0.1 and abs(theta_dot) < 0.5:
            stable_counter += 1
        else:
            stable_counter = 0

        if k % 20 == 0:
            print(
                f"t = {t:5.2f} s, mode = {mode}, "
                f"theta = {theta:+.3f}, theta_dot = {theta_dot:+.3f}"
            )

        if render and env.render_mode == "human":
            env.render()

        obs, _, _, _, _ = env.step(np.array([u], dtype=np.float32))

        if stable_counter >= stable_required_steps:
            print(f"Pole stabilised upright with {label}; stopping episode.")
            break

    return np.array(t_log), np.array(x_log), np.array(theta_log), np.array(u_log)


def run_pole_episode(env, K, label="Pole-placement controller", total_time=16.0, render=True):
    dt = env.tau
    num_steps = int(total_time / dt)

    obs, _ = env.reset()
    use_stabiliser = False

    print(f"\n=== Running episode with controller: {label} ===")

    stable_counter = 0
    stable_required_steps = int(2.0 / dt)  # 2 seconds

    t_log, x_log, theta_log, u_log = [], [], [], []

    for k in range(num_steps):
        t = k * dt
        u, use_stabiliser, mode = hybrid_state_feedback_controller(
            env, obs, K, use_stabiliser
        )

        x, x_dot, theta, theta_dot = obs

        # Logging
        t_log.append(t)
        x_log.append(x)
        theta_log.append(theta)
        u_log.append(u)

        # Stability check
        if use_stabiliser and abs(theta) < 0.1 and abs(theta_dot) < 0.5:
            stable_counter += 1
        else:
            stable_counter = 0

        if k % 20 == 0:
            print(
                f"t = {t:5.2f} s, mode = {mode}, "
                f"theta = {theta:+.3f}, theta_dot = {theta_dot:+.3f}"
            )

        if render and env.render_mode == "human":
            env.render()

        obs, _, _, _, _ = env.step(np.array([u], dtype=np.float32))

        if stable_counter >= stable_required_steps:
            print(f"Pole stabilised upright with {label}; stopping episode.")
            break

    return np.array(t_log), np.array(x_log), np.array(theta_log), np.array(u_log)


# ---------- MAIN: run both controllers + plots ----------

def main():
    # Env with visualization
    env = CartPoleSwingUpEnv(render_mode="human")

    # Linear model around upright -> pole placement gain
    A, B = linearize_cartpole(env)
    K_pp = design_pole_placement(A, B)

    # 1) PID episode (visual + logging)
    t_pid, x_pid, theta_pid, u_pid = run_pid_episode(env, label="PID controller")

    # 2) Pole-placement episode (visual + logging)
    t_pp, x_pp, theta_pp, u_pp = run_pole_episode(
        env, K_pp, label="Pole-placement controller"
    )

    env.close()

    # ---- Plots: PID vs Pole-placement ----

    # Pole angle
    plt.figure()
    plt.plot(t_pid, theta_pid, label="PID")
    plt.plot(t_pp, theta_pp, label="Pole placement")
    plt.xlabel("Time [s]")
    plt.ylabel("Pole angle θ [rad]")
    plt.title("Pole angle: PID vs Pole placement")
    plt.legend()
    plt.grid(True)

    # Cart position
    plt.figure()
    plt.plot(t_pid, x_pid, label="PID")
    plt.plot(t_pp, x_pp, label="Pole placement")
    plt.xlabel("Time [s]")
    plt.ylabel("Cart position x [m]")
    plt.title("Cart position: PID vs Pole placement")
    plt.legend()
    plt.grid(True)

    # Control input
    plt.figure()
    plt.plot(t_pid, u_pid, label="PID")
    plt.plot(t_pp, u_pp, label="Pole placement")
    plt.xlabel("Time [s]")
    plt.ylabel("Control force u [N]")
    plt.title("Control input: PID vs Pole placement")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()

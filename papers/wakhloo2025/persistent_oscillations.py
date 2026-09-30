"""
Reimplementation (not the authors' code) of the model in
Wakhloo, Clark & Abbott (2025), "Associative synaptic plasticity creates
dynamic persistent activity", bioRxiv 10.1101/2025.08.20.671131.

    dx_i/dt   = -x_i + sum_j (J_ij + A_ij) phi(x_j) + h_i(t)           (Eq. 1)
    h_i(t)    = I cos(2 pi f t + theta_i)  for t < 0, else 0          (Eq. 2)
    p dA_ij/dt = -A_ij + (k/N) phi(x_i) phi(x_j)                       (Eq. 3)

J_ij ~ N(0, g^2/N), phi = tanh. Stimulation runs from t = -T_stim to 0,
then the input is removed. Solver: Diffrax Tsit5, adaptive steps,
rtol = atol = 1e-6, as described in the paper's Methods.
"""
import jax
import jax.numpy as jnp
import diffrax
import numpy as np

# Use 64-bit floats. With tolerances of 1e-6, float32 round-off would
# be close to the solver's error budget and could distort long runs.
jax.config.update("jax_enable_x64", True)


def simulate(N=1000, g=1.3, p=25.0, k=2.8, I=0.65, f=0.1,
             T_stim=100.0, T_after=300.0, n_save=2000, seed=0):
    """
    Simulate one network and return its trajectory.

    Parameters
    ----------
    N       : number of neurons.
    g       : gain of the random backbone J. g > 1 makes the network
              chaotic when there is no plasticity.
    p       : synaptic timescale relative to neurons (neurons have tau = 1).
              p > 1 means synapses are slower than neurons.
    k       : plasticity strength. k > 0 is Hebbian, k = 0 turns plasticity off.
    I, f    : amplitude and frequency of the oscillatory input.
    T_stim  : how long the input is on (from t = -T_stim to t = 0).
    T_after : how long to keep simulating after the input is switched off.
    n_save  : number of evenly spaced time points to store.
    seed    : random seed for J, the input phases, and x(0).

    Returns
    -------
    ts : (n_save,) time points.
    x  : (n_save, N) neuron preactivations.
    A  : (n_save, N, N) plastic weight component. This is large:
         n_save * N^2 floats, so lower n_save for big N.
    J  : (N, N) fixed random backbone.
    """
    # Split one random key into independent keys for each random quantity,
    # so changing one doesn't change the others.
    key = jax.random.PRNGKey(seed)
    kJ, kth, kx = jax.random.split(key, 3)

    # Fixed random connectivity with standard deviation g / sqrt(N).
    # This scaling keeps the eigenvalue bulk inside a disk of radius ~g
    # (the circular law), independent of N.
    J = g / jnp.sqrt(N) * jax.random.normal(kJ, (N, N))

    # Each neuron gets its own input phase, drawn uniformly on [0, 2*pi).
    theta = 2 * jnp.pi * jax.random.uniform(kth, (N,))

    # Initial state: random neural activity and no plastic component,
    # matching the paper's initialization (x_i(0) ~ N(0,1), A_ij(0) = 0).
    x0 = jax.random.normal(kx, (N,))
    A0 = jnp.zeros((N, N))

    def field(t, y, args):
        """Right-hand side of Eqs. 1-3. The state y is the pair (x, A)."""
        x, A = y
        r = jnp.tanh(x)  # firing rates phi(x)

        # External input (Eq. 2): oscillatory drive before t = 0, silence after.
        h = jnp.where(t < 0.0, I * jnp.cos(2 * jnp.pi * f * t + theta), 0.0)

        # Neuron dynamics (Eq. 1): leak, recurrent input through the total
        # connectivity J + A, and external input.
        dx = -x + (J + A) @ r + h

        # Synapse dynamics (Eq. 3): A decays toward zero on timescale p and is
        # pushed by the outer product of post- and presynaptic rates.
        # jnp.outer(r, r)[i, j] = r_i * r_j  ("post times pre").
        # The 1/N scaling makes A small relative to J (which scales as 1/sqrt(N)).
        dA = (-A + (k / N) * jnp.outer(r, r)) / p
        return dx, dA

    # Time points at which to record the solution.
    ts = jnp.linspace(-T_stim, T_after, n_save)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),               # Tsitouras 5/4 Runge-Kutta, as in the paper
        t0=-T_stim, t1=T_after,
        dt0=0.05,                      # initial step; the controller adapts it
        y0=(x0, A0),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(
            rtol=1e-6, atol=1e-6,      # tolerances from the paper's Methods
            # The input switches off abruptly at t = 0. Telling the solver about
            # this jump makes it step exactly to t = 0 instead of stepping over
            # the discontinuity and wasting rejected steps.
            jump_ts=jnp.array([0.0]),
        ),
        max_steps=10_000_000,          # generous cap; chaotic runs need many steps
    )

    x, A = sol.ys
    # Convert from JAX arrays to NumPy for analysis and plotting.
    return np.asarray(ts), np.asarray(x), np.asarray(A), np.asarray(J)


def outlier_eigenvalues(J, A_halt, n=4):
    """
    Return the n eigenvalues of J + A(0) with the largest real part.

    A(0) is the plastic component at the moment the input switches off.
    In the persistent-oscillation regime, a complex-conjugate pair of
    eigenvalues is pulled outside the random bulk (radius ~g). Its imaginary
    part sets the oscillation frequency (cf. the paper's Fig. 4).
    """
    ev = np.linalg.eigvals(J + A_halt)
    return ev[np.argsort(-ev.real)][:n]  # sort by real part, descending


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # N = 300 keeps the run fast on a CPU. The paper uses N = 500 to 5000.
    ts, x, A, J = simulate(N=300)

    # Index of the first saved time point at or after t = 0 (input off).
    i_halt = np.searchsorted(ts, 0.0)

    print("Top eigenvalues of J + A(0):", np.round(outlier_eigenvalues(J, A[i_halt]), 3))
    print("Bulk radius ~ g =", 1.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: firing rates of a few example neurons. Persistent oscillations
    # show up as oscillations that continue to the right of the dashed line.
    ax1.plot(ts, np.tanh(x[:, :5]))
    ax1.axvline(0, color="k", ls="--", lw=1)  # moment the input is removed
    ax1.set(xlabel="time", ylabel="phi(x)", title="Example units (input off at t=0)")

    # Right: full spectrum of the connectivity at t = 0. Look for a pair of
    # points (mirror images across the real axis) outside the bulk disk.
    ev = np.linalg.eigvals(J + A[i_halt])
    ax2.scatter(ev.real, ev.imag, s=4)
    ax2.set(xlabel="Re", ylabel="Im", title="Spectrum of J + A(0)", aspect="equal")

    plt.tight_layout()
    plt.savefig("persistent_oscillations.png", dpi=120)
    print("Saved persistent_oscillations.png")
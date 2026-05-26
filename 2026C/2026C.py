# Candidate Number: 1104704

# Question C.1

import numpy as np

# defines a function that performs a singular iteration of the Kac ring
def kac_ring_step(balls, edges):
    # preventing pass-by-reference modification of the original data
    balls, edges = balls.copy(), edges.copy()
    # temporary value for swapping
    prev_value = balls[0]
    # loop through balls
    for i in range(len(balls)):
        # swap colour of ball if it passes over an edge
        if edges[i]:
            prev_value += 1
            prev_value %= 2
        # increase the position of the current ball
        next_position = (i+1)%len(balls)
        prev_value, balls[next_position] = balls[next_position], prev_value

    return (balls, edges)

# Example as in Figure C.1
ring = ([1,1,1,0,0,1,0,0,1,0,0,0], [1,1,0,0,1,0,0,1,0,0,0,1])
new_ring = kac_ring_step(*ring)
print(new_ring)


# Question C.2

import matplotlib.pyplot as plt

# function to generate a visualisation of a Kac ring at a particular time step
def plot_kac_ring(balls, edges, step=None, ax=None):

    N = len(balls)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # Equally spaced sites around the circle
    # orient the ring with ball 0 at the top, going clockwise
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Draw edges
    for i in range(N):
        j = (i + 1) % N

        edge_colour = "red" if edges[i] else "black"
        edge_width = 2.5 if edges[i] else 1.0

        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            color=edge_colour,
            linewidth=edge_width,
            zorder=1,
        )

    # Draw balls
    ball_colours = ["white" if b == 0 else "black" for b in balls]

    ax.scatter(
        x,
        y,
        s=350,
        c=ball_colours,
        edgecolors="black",
        linewidths=1.5,
        zorder=2,
    )

    # labels for each position
    for i in range(N):
        ax.text(
            1.15 * x[i],
            1.15 * y[i],
            str(i),
            ha="center",
            va="center",
            fontsize=9,
        )

    if step is None:
        ax.set_title(f"N = {N}", pad=20)
    else:
        ax.set_title(f"N = {N} step = {step}", pad=20)

    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax

# original Kac ring
fig, ax = plot_kac_ring(*ring, step=0)
plt.show()

# same Kac ring after 1 time step
fig, ax = plot_kac_ring(*new_ring, step=1)
plt.show()


# Question C.3

# initial parameters
N = 12
mu = 1 / 3

# set seed to generate same results every run
rng = np.random.default_rng(seed=1)

# All balls initially white
balls = np.zeros(N, dtype=int)

# Mark each edge independently with probability mu
edges = rng.random(N) < mu

# Plot step 0 and the next three steps
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for step, ax in enumerate(axes):
    plot_kac_ring(balls, edges, step=step, ax=ax)

    if step < 3:
        balls = kac_ring_step(balls, edges)[0]

plt.tight_layout()
plt.show()


# Question C.4

# defines a function to simulate macroscopic evolution of the system
def simulate_ensemble(M, N, mu, T, seed=None):
    rng = np.random.default_rng(seed)

    # balls[m, i] is the colour of the ball at site i in ring m
    # Convention: 0 = white, 1 = black
    balls = np.zeros((M, N), dtype=np.int8)

    # Each ensemble member has independently generated edges
    edges = rng.random((M, N)) < mu

    deltas = np.empty((M, T + 1), dtype=float)

    for t in range(T + 1):
        # B = number of black balls
        B = np.sum(balls, axis=1)

        # $W = N - B$, so $\delta = B - W = 2B - N$
        deltas[:, t] = (2 * B - N) / N

        if t < T:
            # A ball crossing a marked edge changes colour.
            # Since balls are 0/1 and edges are 0/1,
            # this flips the colour exactly when edges is 1.
            crossed_edges = np.logical_xor(balls, edges).astype(np.int8)

            # Move every ball one site clockwise:
            # ball at site i moves to site (i + 1) % N.
            balls = np.roll(crossed_edges, shift=1, axis=1)

    return deltas, edges

# simulate with designated parameters
M = 100
N = 500
mu = 0.009
T = 1000

deltas, edges = simulate_ensemble(M, N, mu, T, seed=1)

t = np.arange(T + 1)

# Macroscopic evolution law:
# $\delta(t) = (1 - 2 \mu)^t \delta(0)$
# Since all balls are initially white, $\delta(0) = -1$.
delta0 = -1
delta_macro = (1 - 2 * mu) ** t * delta0

# Sample mean across the ensemble
delta_mean = np.mean(deltas, axis=0)

plt.figure(figsize=(10, 6))

# Individual trajectories
for m in range(M):
    label = "individual trajectories" if m == 0 else None
    plt.plot(t, deltas[m], color="gray", alpha=0.25, linewidth=0.8, label=label)

# Macroscopic evolution law
plt.plot(
    t,
    delta_macro,
    color="black",
    linewidth=2.5,
    label=r"macroscopic law $(1 - 2\mu)^t \delta(0)$",
)

# Sample mean
plt.plot(
    t,
    delta_mean,
    color="red",
    linewidth=2,
    linestyle="--",
    label="sample mean",
)

plt.xlabel(r"time $t$")
plt.ylabel(r"$\delta(t) = \Delta(t)/N$")
plt.title(r"Kac ring ensemble: $M = 100$, $N = 500$, $\mu = 0.009$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Question C.5

# At $t=500$, $\delta(t) = \pm 1$.
# This is because $t=N$, and thus every ball has completed a circuit around the ring.
# This means every ball has changed colour $n$ times, where $n$ is the number of marked edges in the ring.
# If n is even, the ring is all white, and $\delta(t) = 1$.
# If n is odd, the ring is all black, and $\delta(t) = 1$.
# This difference in behaviour is a consequence of the reversible microscopic, rather than the irreversible macroscopic, laws of the system.


# Question C.6
def simulate_ensemble(M, N, mu, T, seed=None, keep=100):
    rng = np.random.default_rng(seed)

    keep = min(keep, M)

    # balls[m, i] is the colour at site i in ring m.
    # False = white, True = black.
    balls = np.zeros((M, N), dtype=bool)

    # edges[m, i] marks edge i -> i + 1 mod N.
    edges = rng.random((M, N)) < mu

    # Work buffer for the updated state.
    new_balls = np.empty_like(balls)

    t = np.arange(T + 1)
    sample_mean = np.empty(T + 1, dtype=np.float64)
    trajectories = np.empty((keep, T + 1), dtype=np.float64)

    for step in range(T + 1):
        # Number of black balls in each ring.
        B = balls.sum(axis=1)

        # delta = (B - W) / N = (2B - N) / N
        deltas = (2.0 * B - N) / N

        sample_mean[step] = deltas.mean()
        trajectories[:, step] = deltas[:keep]

        if step == T:
            break

        np.logical_xor(
            balls[:, :-1],
            edges[:, :-1],
            out=new_balls[:, 1:]
        )

        # site 0 receives balls from site N-1
        np.logical_xor(
            balls[:, -1],
            edges[:, -1],
            out=new_balls[:, 0]
        )

        # Swap buffers.
        balls, new_balls = new_balls, balls

    return t, sample_mean, trajectories, edges

M = 1000
Ns = [100, 800, 16000, 32000, 64000]

results = {}

for N in Ns:
    mu = 1 / np.sqrt(N)
    T = N // 2

    print(f"Simulating N = {N}, mu = {mu:.6f}, T = {T}")

    t, sample_mean, trajectories, edges = simulate_ensemble(
        M=M,
        N=N,
        mu=mu,
        T=T,
        seed=12345 + N,
        keep=100,
    )

    # Since all balls are initially white, delta(0) = -1.
    delta0 = -1.0

    # Macroscopic evolution law:
    # delta(t) = (1 - 2mu)^t delta(0)
    delta_macro = (1 - 2 * mu) ** t * delta0

    # Variance bound from the question:
    # Var[delta(t)] <= (1/N) * (1/(2mu(1 - mu)) - 1)
    variance_bound = (1 / N) * (1 / (2 * mu * (1 - mu)) - 1)
    std_bound = np.sqrt(variance_bound)

    results[N] = {
        "t": t,
        "mu": mu,
        "sample_mean": sample_mean,
        "trajectories": trajectories,
        "delta_macro": delta_macro,
        "std_bound": std_bound,
    }

fig, axes = plt.subplots(
    len(Ns),
    1,
    figsize=(10, 18),
    sharey=True,
    constrained_layout=True,
)

for ax, N in zip(axes, Ns):
    result = results[N]

    t = result["t"]
    mu = result["mu"]
    sample_mean = result["sample_mean"]
    trajectories = result["trajectories"]
    delta_macro = result["delta_macro"]
    std_bound = result["std_bound"]

    # First 100 trajectories
    for k in range(trajectories.shape[0]):
        ax.plot(
            t,
            trajectories[k],
            color="gray",
            alpha=0.18,
            linewidth=0.7,
        )

    # Macroscopic law
    ax.plot(
        t,
        delta_macro,
        color="black",
        linewidth=2.0,
        label="macroscopic law",
    )

    # Macroscopic law ± standard deviation bound
    ax.plot(
        t,
        delta_macro + std_bound,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=r"macroscopic law $\pm$ std. bound",
    )

    ax.plot(
        t,
        delta_macro - std_bound,
        color="black",
        linestyle=":",
        linewidth=1.5,
    )

    # Sample mean
    ax.plot(
        t,
        sample_mean,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="sample mean",
    )

    ax.set_title(
        rf"$N = {N}$, $M = {M}$, $\mu = 1/\sqrt{{N}} = {mu:.5f}$"
    )
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\delta(t)$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

plt.show()


# Question C.7

import numpy as np
import matplotlib.pyplot as plt

# compute x log(x) for an array
def xlogx(x):
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)

    mask = x > 0
    out[mask] = x[mask] * np.log(x[mask])

    return out

# entropy per position for the Kac ring as a function of delta
def entropy_per_site(delta):
    delta = np.asarray(delta)

    return (
        np.log(2)
        - 0.5 * (
            xlogx(1 + delta)
            + xlogx(1 - delta)
        )
    )

# create plot
delta = np.linspace(-1, 1, 1000)
s = entropy_per_site(delta)

imax = np.argmax(s)
delta_max = delta[imax]
s_max = s[imax]

plt.figure(figsize=(8, 5))
plt.plot(delta, s, linewidth=2)
plt.scatter(delta_max, s_max, color="red", zorder=3)

plt.xlabel(r"$\delta$")
plt.ylabel(r"$s(\delta)$")
plt.title(r"Entropy per site for the Kac ring")
plt.grid(True, alpha=0.3)

plt.annotate(
    rf"maximum at $\delta \approx {delta_max:.3f}$",
    xy=(delta_max, s_max),
    xytext=(0.1, s_max - 0.15),
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()
plt.show()

print(f"Maximum entropy occurs at delta ≈ {delta_max:.6f}")
print(f"Maximum entropy value is s ≈ {s_max:.6f}")


# Question C.8

def delta_of_balls(balls):
    balls = np.asarray(balls, dtype=int).ravel()

    N = len(balls)
    B = np.sum(balls == 1)
    W = N - B

    return (B - W) / N

N = 12
T = 4

# initial conditions
balls = np.array(
    [1, 0, 1, 0, 0, 1,
     0, 0, 1, 1, 1, 0],
    dtype=int,
)

markers = np.array(
    [0, 1, 0, 1, 0, 1,
     0, 0, 1, 1, 1, 1],
    dtype=int,
)

states = []
deltas = []
entropies = []

# do entropy calculations during simulation
for t in range(T + 1):
    states.append(balls.copy())

    delta = delta_of_balls(balls)
    entropy = entropy_per_site(delta)

    deltas.append(delta)
    entropies.append(entropy)

    if t < T:
        balls, edges = kac_ring_step(balls, markers)

deltas = np.array(deltas)
entropies = np.array(entropies)

# truncate and tabularise results
print(" t    delta(t)      s(delta(t))")
print("--------------------------------")
for t, delta, entropy in zip(range(T + 1), deltas, entropies):
    print(f"{t:2d}   {delta: .6f}     {entropy: .6f}")

# plot configurations using code from C.2
plt.figure(figsize=(7, 4), constrained_layout=True)

plt.plot(range(T + 1), entropies, marker="o", linewidth=2)

plt.xlabel(r"time $t$")
plt.ylabel(r"entropy per site $s$")
plt.title("Entropy decrease from a special maximal-entropy initial condition")
plt.grid(True, alpha=0.3)

plt.show()
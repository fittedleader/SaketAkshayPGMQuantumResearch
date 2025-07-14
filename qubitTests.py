import numpy as np
import cvxpy as cp
from numpy.linalg import qr

def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1: Generate N×N matrix Z with complex numbers a+bi where both a and b 
    # are normally distributed with mean 0 and variance 1 (Ginibre ensemble)
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2: Compute QR decomposition Z = QR
    Q, R = qr(Z)

    # Step 3: Compute diagonal matrix Λ = diag(R_ii/|R_ii|)
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4: Compute Q' = QΛ, which will be Haar-random
    return np.dot(Q, Lambda)

def generate_haar_random_state(N):
    """Generate a Haar-random quantum state vector of dimension N."""
    # Generate Haar-random unitary matrix
    U = qr_haar(N)
    
    # Apply to a fixed basis state (e.g., |0⟩) to get random state
    # |0⟩ = [1, 0, 0, ..., 0]^T
    basis_state = np.zeros(N, dtype=complex)
    basis_state[0] = 1.0
    
    # Apply the Haar-random unitary to get a random state
    random_state = U @ basis_state
    
    return random_state

def trace_norm(A):
    """Calculate the trace norm of matrix A."""
    # Trace norm is the sum of singular values
    
    # For 1x1 matrices, the trace norm is just the absolute value
    if A.shape == (1, 1):
        return np.abs(A[0, 0])
    
    # For larger matrices, use SVD
    singular_values = np.linalg.svd(A, compute_uv=False)
    return np.sum(singular_values)

def helstrom_bound(rho1, rho2):
    """Calculate the Helstrom bound for minimum error discrimination between two states."""
    # Helstrom bound: 1/2 + 1/4 * trace_norm(rho1 - rho2)
    difference = rho1 - rho2
    trace_norm_val = trace_norm(difference)
    optimal_probability = 0.5 + 0.25 * trace_norm_val
    return optimal_probability

def helstrom_bound_pure(psi1, psi2):
    """
    Calculate the Helstrom bound for two pure states |psi1> and |psi2>.
    """
    inner_product = np.vdot(psi1, psi2)
    modulus_squared = np.abs(inner_product) ** 2
    helstrom = 0.5 + 0.5 * np.sqrt(1 - modulus_squared)
    return helstrom

def matrix_power(A, power):
    """Compute matrix power using eigenvalue decomposition."""
    eigenvals, eigenvecs = np.linalg.eigh(A)
    # Ensure eigenvalues are positive (for numerical stability)
    eigenvals = np.maximum(eigenvals, 1e-12)
    # Compute power
    eigenvals_power = eigenvals ** power
    # Reconstruct matrix
    return eigenvecs @ np.diag(eigenvals_power) @ eigenvecs.conj().T

def compute_inv_sqrt(S, eps=1e-10):
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ inv_sqrt_diag @ eigvecs.conj().T

def pretty_good_measurement(density_matrices, priors, eps=1e-10):
    M = len(density_matrices)
    N = density_matrices[0].shape[0]
    S = np.zeros((N, N), dtype=complex)
    for i in range(M):
        S += priors[i] * density_matrices[i]
    S = (S + S.conj().T) / 2
    S_inv_half = compute_inv_sqrt(S, eps)
    povm_elements = []
    for i in range(M):
        E_i = S_inv_half @ (priors[i] * density_matrices[i]) @ S_inv_half
        E_i = (E_i + E_i.conj().T) / 2
        # Project onto PSD cone
        eigvals, eigvecs = np.linalg.eigh(E_i)
        eigvals = np.maximum(eigvals, 0)
        E_i_psd = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        povm_elements.append(E_i_psd)
    return povm_elements

def calculate_pgm_probability(povm_elements, pure_states, priors):
    probability = 0
    for i in range(len(pure_states)):
        psi = pure_states[i].reshape(-1, 1)
        sigma = psi @ psi.T.conj()
        probability += priors[i] * np.real(np.trace(povm_elements[i] @ sigma))
    return probability

def solve_sdp_optimal(density_matrices, M, N):
    """Solve SDP for optimal POVM when M > 2."""
    # Create POVM variables
    povm_elements = []
    for i in range(M):
        P = cp.Variable((N, N), complex=True)
        povm_elements.append(P)

    # Set up constraints
    constraints = []

    # POVM elements must be Hermitian and PSD
    for P in povm_elements:
        constraints.append(P >> 0)  # PSD
        constraints.append(P == P.H)  # Hermitian

    # Sum of POVM elements equals identity
    constraints.append(sum(povm_elements) == np.eye(N))

    # Set up objective function
    objective = 0
    for i in range(M):
        objective += cp.real(cp.trace(povm_elements[i] @ density_matrices[i]))
    objective = objective / M

    # Solve the SDP
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.SCS, eps=1e-10, max_iters=50000, alpha=1.5)

    # Post-process to ensure PSD
    optimal_povms = []
    for P in povm_elements:
        P_val = P.value
        # Ensure Hermiticity
        P_val = (P_val + P_val.conj().T) / 2
        # Project onto PSD cone
        eigenvals, eigenvecs = np.linalg.eigh(P_val)
        eigenvals = np.maximum(eigenvals, 0)  # Set negative eigenvalues to zero
        P_val = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        optimal_povms.append(P_val)

    # Recalculate objective with post-processed POVMs
    final_objective = 0
    for i in range(M):
        final_objective += np.real(np.trace(optimal_povms[i] @ density_matrices[i]))
    final_objective = final_objective / M

    return final_objective, optimal_povms

def pgm_lower_bound(M, P_best):
    return 1/M + ((1 - M*P_best)**2) / (M*(M-1)) if M > 1 else P_best

print("=" * 60)
print("Qubit Tests: Quantum State Discrimination")
print("Using Haar-Random Quantum States")
print("=" * 60)
print()

# Get user input for dimension and number of states
while True:
    try:
        N = int(input("Enter the dimension N (>=1): "))
        if N >= 1:
            break
        else:
            print("Please enter an integer >= 1.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

while True:
    try:
        M = int(input("Enter the number of states M (>=2): "))
        if M >= 2:
            break
        else:
            print("Please enter an integer >= 2.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

print()
print(f"Dimension: N = {N}")
print(f"Number of states: M = {M}")
print()

# Get number of trials from user
while True:
    try:
        num_trials = int(input("Enter the number of trials: "))
        if num_trials > 0:
            break
        else:
            print("Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Please enter a positive integer.")

print()
print(f"Running {num_trials} trials...")
print()

# Store results for all trials
helstrom_probabilities = []
pgm_probabilities = []
sdp_probabilities = []

pgm_lower_bounds = []
pgm_minus_lb_diffs = []
helstrom_minus_pgm_diffs = []

for trial in range(num_trials):
    print(f"Trial {trial + 1}/{num_trials}")
    # Generate Haar-random quantum states
    states = []
    for i in range(M):
        state = generate_haar_random_state(N)
        states.append(state.flatten())  # Convert to 1D array for consistency

    # Convert states to density matrices
    density_matrices = []
    for state in states:
        rho = np.outer(state, np.conj(state))
        density_matrices.append(rho)

    priors = [1.0 / M] * M
    if M == 2:
        psi1, psi2 = states[0], states[1]
        helstrom_prob = helstrom_bound_pure(psi1, psi2)
        pgm_elements = pretty_good_measurement(density_matrices, priors)
        pgm_prob = calculate_pgm_probability(pgm_elements, [states[0], states[1]], priors)
        pgm_lb = pgm_lower_bound(M, helstrom_prob)
        pgm_lower_bounds.append(pgm_lb)
        pgm_minus_lb_diffs.append(pgm_prob - pgm_lb)
        helstrom_minus_pgm_diffs.append(helstrom_prob - pgm_prob)
        helstrom_probabilities.append(helstrom_prob)
        pgm_probabilities.append(pgm_prob)
    else:
        pgm_elements = pretty_good_measurement(density_matrices, priors)
        pgm_prob = calculate_pgm_probability(pgm_elements, states, priors)
        sdp_prob, optimal_povms = solve_sdp_optimal(density_matrices, M, N)
        pgm_lb = pgm_lower_bound(M, sdp_prob)
        pgm_lower_bounds.append(pgm_lb)
        pgm_minus_lb_diffs.append(pgm_prob - pgm_lb)
        helstrom_minus_pgm_diffs.append(sdp_prob - pgm_prob)
        sdp_probabilities.append(sdp_prob)
        pgm_probabilities.append(pgm_prob)

# Summary statistics
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Number of trials: {num_trials}")
print(f"Number of states: M = {M}")
print(f"Dimension: N = {N}")

if M == 2:
    print("Helstrom bounds:")
    print(np.array(helstrom_probabilities))
    print(f"Average Helstrom bound: {np.nanmean(helstrom_probabilities):.6f}")
    print()
    print("PGM probabilities:")
    print(np.array(pgm_probabilities))
    print(f"Average PGM probability: {np.nanmean(pgm_probabilities):.6f}")
    print()
    print("PGM lower bounds:")
    print(np.array(pgm_lower_bounds))
    print(f"Average PGM lower bound: {np.nanmean(pgm_lower_bounds):.6f}")
    print()
    print("=" * 40)
    print("Helstrom - PGM differences:")
    print(np.array(helstrom_minus_pgm_diffs))
    print()
    print("PGM - PGM lower bound differences:")
    print(np.array(pgm_minus_lb_diffs))
    print()
    print(f"Average (PGM - PGM lower bound): {np.nanmean(pgm_minus_lb_diffs):.6f}")
    print(f"Max (PGM - PGM lower bound): {np.nanmax(pgm_minus_lb_diffs):.6f}")
    print(f"Min (PGM - PGM lower bound): {np.nanmin(pgm_minus_lb_diffs):.6f}")
else:
    print("SDP optimal bounds:")
    print(np.array(sdp_probabilities))
    print(f"Average SDP optimal probability: {np.nanmean(sdp_probabilities):.6f}")
    print()
    print("PGM probabilities:")
    print(np.array(pgm_probabilities))
    print(f"Average PGM probability: {np.nanmean(pgm_probabilities):.6f}")
    print()
    print("PGM lower bounds:")
    print(np.array(pgm_lower_bounds))
    print(f"Average PGM lower bound: {np.nanmean(pgm_lower_bounds):.6f}")
    print()
    print("=" * 40)
    print("SDP - PGM differences:")
    print(np.array(helstrom_minus_pgm_diffs))
    print()
    print("PGM - PGM lower bound differences:")
    print(np.array(pgm_minus_lb_diffs))
    print()
    print(f"Average (PGM - PGM lower bound): {np.nanmean(pgm_minus_lb_diffs):.6f}")
    print(f"Max (PGM - PGM lower bound): {np.nanmax(pgm_minus_lb_diffs):.6f}")
    print(f"Min (PGM - PGM lower bound): {np.nanmin(pgm_minus_lb_diffs):.6f}")

print()
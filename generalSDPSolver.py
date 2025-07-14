import numpy as np
import cvxpy as cp
from qutipy.general_functions import ket
from qutipy.states import random_state_vector
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

print("=" * 60)
print("Optimal POVM for Minimum Error Discrimination")
print("Using Haar-Random Quantum States")
print("=" * 60)
print()

# Get user input for dimension and number of states
while True:
    try:
        N = int(input("Enter the dimension N (>=2): "))
        if N >= 2:
            break
        else:
            print("Please enter an integer >= 2.")
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

# Generate Haar-random quantum states
print("Generating Haar-random quantum states...")
states = []
for i in range(M):
    state = generate_haar_random_state(N)
    states.append(state.flatten())  # Convert to 1D array for consistency

# Convert states to density matrices
density_matrices = []
for state in states:
    rho = np.outer(state, np.conj(state))
    density_matrices.append(rho)

# Display the states
print("Generated Haar-random quantum states:")
for i, state in enumerate(states, 1):
    print(f"|ψ_{i}⟩ = {state}")
print()

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
print("Solving SDP for optimal POVM...")
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

print()
print("=" * 40)
print("RESULTS")
print("=" * 40)
print(f"Optimal success probability: {final_objective:.6f}")
print()

# Verify constraints
print("=" * 40)
print("CONSTRAINT VERIFICATION")
print("=" * 40)

# Check POVM sum equals identity
povm_sum = sum(optimal_povms)
print(f"POVM sum equals identity: {np.allclose(povm_sum, np.eye(N), atol=1e-10)}")

# Check each POVM element is PSD
all_psd = True
for i, P in enumerate(optimal_povms):
    eigenvals = np.linalg.eigvals(P)
    is_psd = np.all(eigenvals >= -1e-12)
    if not is_psd:
        all_psd = False
    print(f"P_{i+1} is PSD: {is_psd}")

# Check Hermiticity
all_hermitian = True
for i, P in enumerate(optimal_povms):
    is_hermitian = np.allclose(P, P.conj().T, atol=1e-10)
    if not is_hermitian:
        all_hermitian = False
    print(f"P_{i+1} is Hermitian: {is_hermitian}")

print(f"\nAll constraints satisfied: {all_psd and all_hermitian}")
print()

# Calculate objective value from optimal POVMs for verification
calculated_objective = 0
for i in range(M):
    calculated_objective += np.real(np.trace(optimal_povms[i] @ density_matrices[i]))
calculated_objective = calculated_objective / M

print("=" * 40)
print("OBJECTIVE VERIFICATION")
print("=" * 40)
print(f"Objective value from optimal POVMs: {calculated_objective:.6f}")
print(f"Objective value from SDP solver: {objective.value:.6f}")
print(f"Difference: {np.abs(calculated_objective - objective.value):.2e}")
print()



print("=" * 60)
print("The optimal value represents the maximum probability of")
print("correctly distinguishing between the quantum states.")
print("This is the Helstrom bound for minimum error discrimination.")
print("=" * 60)


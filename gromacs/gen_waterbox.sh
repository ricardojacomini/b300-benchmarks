#!/bin/bash
# Generate water-box benchmark TPR during Docker image build
# Called by Dockerfile — no external downloads needed
set -e

TOPDIR=/usr/local/gromacs/share/gromacs/top
BENCHDIR=/benchmarks
cd "${BENCHDIR}"

# Step 1: Initial topology (gmx solvate will append SOL count)
cat > init.top << 'TOP'
#include "gromos43a1.ff/forcefield.itp"
#include "gromos43a1.ff/spc.itp"

[ system ]
Water benchmark (9x9x9 nm, SPC, gromos43a1)

[ molecules ]
TOP

# Step 2: Create 9x9x9 nm SPC water box (~23,905 molecules, ~71k atoms)
gmx solvate -cs "${TOPDIR}/spc216.gro" -box 9 9 9 -o waterbox.gro -p init.top 2>&1

# Step 3: MD parameters — production run settings
cat > water.mdp << 'MDP'
integrator      = md
dt              = 0.002
nsteps          = 50000
nstlog          = 5000
nstxout         = 0
nstvout         = 0
nstenergy       = 5000
cutoff-scheme   = Verlet
nstlist         = 40
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
DispCorr        = EnerPres
tcoupl          = V-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = 300
pcoupl          = no
pbc             = xyz
MDP

# Step 4: Generate the run input (TPR)
gmx grompp -f water.mdp -c waterbox.gro -p init.top -o water_bench.tpr -maxwarn 3 2>&1

echo "Water-box TPR ready: ${BENCHDIR}/water_bench.tpr"
ls -lh "${BENCHDIR}/water_bench.tpr"

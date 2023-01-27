#!/usr/bin/bash

#SBATCH --job-name MyJob
#SBATCH --comment "No one will read this anyway..."

#SBATCH --mail-type=ALL
#SBATCH --mail-user=Reka.schwengelbeck@physik.uni-muenchen.de


python3 ed_extended_hubbard_mesoscopic_leads.py
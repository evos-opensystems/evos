#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:17:49 2023

@author: reka
"""

nubig normales tdvp paper
2014 "uto" lubig, hagemann 
mps related dirac frenkl principle 
review mps zeitentwicklung von sebastian 



##########################################################################################################
updateschema
single site  update schema
- two site update schema: bond dimension kann wachsen- langsam für bosonische oder komplizizierte Systeme 


lokale subspace expansion

GSE:
kleine bond dimension: globale subspace expansion
single site schema mit bond dimension kann wachsen
krylov subspace expansion

modi wechsel während zeitentwicklung 

keine neue konfig 

tdvp.set_tdvp_mode(p.tdvp.Mode.Subspace)
tdvp = p.mp.tdvp.PTDVP(initial_state,Hs,conf)
###########################################################################################################

bevor MPS
- mattia
- mesoleads als methode abstrahieren
angeben
anzahl bad  sites
spektraldichte bad sites 
temperaur ==> hamiltonian leads kopplungen 
und dann rechnet er einfach runter
- globale gse parameters



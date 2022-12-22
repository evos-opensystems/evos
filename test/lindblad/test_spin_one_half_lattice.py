import evos
import evos.src.lattice.lattice as lat
import evos.src.methods.lindblad as lindblad
import numpy as np
n_sites=1

spin_lat_test = lat.Lattice('ed')
spin_lat_test.specify_lattice('spin_one_half_lattice')
spin_lat_test = spin_lat_test.spin_one_half_lattice.SpinOneHalfLattice(n_sites)
sp_0 = spin_lat_test.sso('sp',0) #sp is annihilator,  sm is creator
sz_0 = spin_lat_test.sso('sz',0)
#sz_1 = spin_lat_test.sso('sz',1)
sx_0 = spin_lat_test.sso('sx',0)
#sx_1 = spin_lat_test.sso('sx',1)
#help(spin_lat_test)
vac = spin_lat_test.vacuum_state #vacuum state = all up
all_down = vac.copy()
all_down = np.dot(sx_0, all_down)
#all_down = np.dot(sx_1, all_down)

#print(np.dot(sp_0,vac))

#exp = np.dot(np.conjugate(vac), np.dot(sz_1,vac))
exp = np.dot(np.conjugate(all_down), np.dot(sz_0,all_down))

#print('exp={0}'.format(exp))

H = sz_0.copy()
lindblad_test = evos.src.methods.lindblad.Lindblad([],H,n_sites,)

#vac_projector = lindblad_test.ket_to_projector(vac)
#print('')
sup_state = 1./np.sqrt(1 + 4)*(all_down + 2*vac)

sup_projector = lindblad_test.ket_to_projector(sup_state)
#print('sup_projector= {0}'.format(sup_projector))

sup_projector_v = lindblad_test.vectorize_density_matrix(sup_projector)
#print(sup_projector_v)
sup_projector_uv = lindblad_test.un_vectorize_density_matrix(sup_projector_v)
#print(sup_projector_uv)
rhs_test = lindblad_test.right_hand_side_lindblad_eq(0, sup_projector_v)

rho_t = lindblad_test.solve_lindblad_equation(sup_projector, 0.1, 1)
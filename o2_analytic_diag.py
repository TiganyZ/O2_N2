# /usr/env/bin/python3.6
##    This is a file which has the analytic solution to the tight binding Hamiltonian of O2   ##
import numpy as np 
import matplotlib.pyplot as plt
"""
This is the Hamiltonian for diatomic Oxygen 
H = np.array( [  [ s,   ss,   0,   0,   0,   -sp, -sp, -sp ], 
			     [ ss,   s,   sp,  sp,  sp,  0,   0,   0   ], 
			     [ 0,   sp,  p,   0,   0,   pps, 0,   0   ],
			     [ 0,   sp,  0,   p,   0,   0,   ppp, 0   ], 
			     [ 0,   sp,  0,   0,   p,   0,   0,   ppp ], 
			     [ -sp, 0,   pps, 0,   0,   p,   0,   0   ], 
			     [ -sp, 0,   0,   ppp, 0,   0,   p,   0   ],
			     [ -sp, 0,   0,   0,   ppp, 0,   0,   p   ]  ]  )
"""

def gsp( V, r0, r, rc, n, nc ):
	return V * (r0 / r) * n * np.exp(  n * (   - ( r / rc ) * nc   +   ( r0 / rc ) * nc   )  )

def epl( A, r0, r, m, p ):
	ret = 0
	for i in range(len(A)):
		ret += A[i] * ( ( r0 / r ) ** m[i]  ) * np.exp( - p[i] * ( r  - r0 ) )
	return ret

def eig_exp1( s, p, ss, sp, pps, ppp):
	##  This obtains the third, fourth and fifth eigenvalues of the above matrix
	p0 = 1.

	p1 = ( -2 * p  +  ppp  +  pps  -  s  -  ss )

	p2 = (p**2   -   p * ppp   -   p * pps   +   ppp * pps   +   2 * p * s   -   ppp * s - 
    			pps * s   -   3 * sp**2   +   2 * p * ss   -   ppp * ss   -   pps * ss  )

	p3 = (- p**2 * s   +   p * ppp * s   +   p * pps * s   -   ppp * pps * s   +   3 * p * sp**2   -   ppp * sp**2 - 
 				2 * pps * sp**2   -   p**2 * ss   +   p * ppp * ss   +   p * pps * ss - 
 				ppp * pps * ss )  

	return np.roots( [p0, p1, p2, p3] )



def eig_exp2( s, p, ss, sp, pps, ppp ):
	##  This obtains the sixth, seventh and eighth eigenvalues of the above matrix
	p0 = 1.

	p1 = ( -2 * p   -   ppp  -  pps  -  s  +  ss )

	p2 = (p**2   +   p * ppp   +   p * pps   +   ppp * pps   +   2 * p * s   +   ppp * s   + 
    			pps * s   -   3 * sp**2   -   2 * p * ss   -   ppp * ss   -   pps * ss )

	p3 = (- p**2 * s   -   p * ppp * s   -   p * pps * s   -   ppp * pps * s   +   3 * p * sp**2   +   ppp * sp**2   + 
    			2 * pps * sp**2   +   p**2 * ss   +   p * ppp * ss   +   p * pps * ss + 
 				ppp * pps * ss )

	return np.roots( [p0, p1, p2, p3] )

def get_O2_eigs(s, p, ss, sp, pps, ppp):
	
	e = np.zeros(8)

	e[0] = p - ppp
	e[1] = p + ppp
    
	ev1  = eig_exp1( s, p, ss, sp, pps, ppp )	
	e[2] = ev1[0]
	e[3] = ev1[1]
	e[4] = ev1[2]

	ev2  = eig_exp2( s, p, ss, sp, pps, ppp )	
	e[5] = ev1[0]
	e[6] = ev2[1]
	e[7] = ev2[2]

	se = np.sum( np.sort(e)[:int(len(e)/2)] )
	return e, se

def get_O2_eigs_simple(s, p, ss, sp, pps, ppp):
	
	e = np.zeros(8)

	e[0] = p - ppp
	e[1] = p - ppp
	e[2] = p + ppp
	e[3] = p + ppp

	e[4] = 0.5 * (p - pps + s + ss - np.sqrt( (p - pps + s + ss)**2 +  4 * ( -p * s  +  pps * s  +  sp**2  -  p * ss + pps * ss) ) )
	e[5] = 0.5 * (p - pps + s + ss + np.sqrt( (p - pps + s + ss)**2 +  4 * ( -p * s  +  pps * s  +  sp**2  -  p * ss + pps * ss) ) )
	e[6] = 0.5 * (p + pps + s - ss - np.sqrt( (p + pps + s - ss)**2 +  4 * ( -p * s  -  pps * s  +  sp**2  +  p * ss + pps * ss) ) )
	e[7] = 0.5 * (p + pps + s - ss + np.sqrt( (p + pps + s - ss)**2 +  4 * ( -p * s  -  pps * s  +  sp**2  +  p * ss + pps * ss) ) )

	se = np.sum( np.sort(e)[:int(len(e)/2)] )

	return e, se


def get_O2_eigs_dd( s, p, ss, sp, pps, ppp ):

	H = np.array( [  [ s,    ss,   0,    0,    0,    -sp,  -sp,  -sp ], 
				     [ ss,    s,   sp,   sp,   sp,   0,    0,    0   ], 
				     [ 0,    sp,   p,    0,    0,    pps,  0,    0   ],
				     [ 0,    sp,   0,    p,    0,    0,    ppp,  0   ], 
				     [ 0,    sp,   0,    0,    p,    0,    0,    ppp ], 
				     [ -sp,  0,    pps,  0,    0,    p,    0,    0   ], 
				     [ -sp,  0,    0,    ppp,  0,    0,    p,    0   ],
				     [ -sp,  0,    0,    0,    ppp,  0,    0,    p   ]  ]  )

	H = np.array( [  [ s,    ss,   0,    0,    0,    -sp,  0,    0 ], 
				     [ ss,    s,   sp,   0,    0,    0,    0,    0   ], 
				     [ 0,    sp,   p,    0,    0,    pps,  0,    0   ],
				     [ 0,     0,   0,    p,    0,    0,    ppp,  0   ], 
				     [ 0,     0,   0,    0,    p,    0,    0,    ppp ], 
				     [ -sp,   0,    pps,  0,    0,    p,    0,    0   ], 
				     [ 0,     0,    0,    ppp,  0,    0,    p,    0   ],
				     [ 0,     0,    0,    0,    ppp,  0,    0,    p   ]  ]  )





	e = np.linalg.eigh(H)[0]
	srt = np.sort(e)

	se = np.sum( np.sort(e)[: int(len(e)/2)] )
	#e[-1] = -e[-1]
	print(se)
	return e, se

def get_O2_eigs_full( s, p, Ess, Esp, Exx, Exy ):

	H = np.array( [  [ s,    Ess,  		0,        0,          0,    -Esp[0],    -Esp[1],   -Esp[2]   ], 

				     [ Ess,   s,      Esp[0],   Esp[1],   Esp[2],      0,           0,        0      ], 

				     [ 0,    Esp[0],    p,        0,       0,        Exx[0],  	 Exy[0],    Exy[1]   ],

				     [ 0,    Esp[1],    0,        p,       0,        Exy[0],     Exx[1],    Exy[2]   ], 

				     [ 0,    Esp[2],    0,        0,       p,        Exy[1],     Exy[2],    Exx[2]   ], 

				     [ -Esp[0],   0,  Exx[0],  	Exy[0],   Exy[1],      p,          0,         0      ], 

				     [ -Esp[1],   0,  Exy[0],   Exx[1],   Exy[2],      0,          p,         0      ],

				     [ -Esp[2],   0,  Exy[1],   Exy[2],   Exx[2],      0,          0,         p      ]  ]  )





	e = np.linalg.eigh(H)[0]
	srt = np.sort(e)

	se = np.sum( np.sort(e)[: int(len(e)/2)] )
	#e[-1] = -e[-1]
	print(se)
	return e, se


def get_matrix_elements( l, m, n,   s, p, ss, sps, pps, ppp ):

	Exx = np.zeros( 3 )
	Exy = np.zeros( 3 )
	Esp = np.zeros( 3 )

	Ess    = ss

	Esp[0] = l * sps
	Esp[1] = m * sps
	Esp[2] = n * sps

	Exx[0] = l**2 * pps  +  ( 1. - l**2  ) * ppp  # xx
	Exx[1] = m**2 * pps  +  ( 1. - m**2  ) * ppp  # yy
	Exx[2] = n**2 * pps  +  ( 1. - n**2  ) * ppp  # zz

	Exy[0] =  l * m * pps   -   l * m * ppp  # xy
	Exy[1] =  l * n * pps   -   l * n * ppp  # xz
	Exy[2] =  m * n * pps   -   m * n * ppp  # yz

	return Ess, Esp, Exx, Exy


def get_bond_integrals(x, l, m, n, DD):
	
	s    = - 2.1164
	p    = - 1.1492

	sss  = - 0.0150
	sps  =   0.0020

	pps  =   0.0500
	ppp  = - 0.0200

	nsss =   2
	nsps =   2

	npps =   3
	nppp =   3

	nc   =   6

	r0   =   5.6 
	rc   =   9.0

	ss  = gsp( sss, r0, x, rc, nsss, nc )
	sps = gsp( sps, r0, x, rc, nsps, nc )
	pps = gsp( pps, r0, x, rc, npps, nc )
	ppp = gsp( ppp, r0, x, rc, nppp, nc )

	if DD == 'Full':
		Ess, Esp, Exx, Exy = get_matrix_elements( l, m, n,   s, p, ss, sps, pps, ppp ) 

		eigs, se = get_O2_eigs_full( s, p, Ess, Esp, Exx, Exy )

	if DD:
		eigs, se = get_O2_eigs_dd(s, p, ss, sps, pps, ppp)

	else:
		eigs, se = get_O2_eigs(s, p, ss, sps, pps, ppp)

	
	return eigs, se



def bond_int_dep(xv, DD, l, m, n):

	e_arr = np.zeros( (8, len(xv)) )

	for i in range(len(xv)):
		eigs = get_bond_integrals( xv[i], DD, l, m, n )
		e_arr[:,i] = eigs

	sbi = np.sum(e_arr , axis = 0) 


	fig = plt.figure()
	ax = fig.add_subplot(111)

	
	#print(sbi)
	#if DD:
	#	sbi = np.sum(e_arr[:-1,:] , axis = 0) - e_arr[-1,:] 
	names = [ 's1', 's2', 'px1', 'py1', 'pz1', 'px2', 'py2', 'pz2' , 'sum'  ]
	for j in range(8):
		ax.plot( xv , e_arr[j,:], label = names[j], linestyle = '-' )

	ax.plot(xv, sbi, label = names[-1], linestyle = '-')
	if DD:
		ax.set_title( 'Bond integrals for diatomic oxygen: DD, -ve last eig' )
	else:
		ax.set_title( 'Bond integrals for diatomic oxygen: Analytic' )

	ax.set_xlabel( 'x (bohr)' )
	ax.set_ylabel( 'Energy (ryd)' )
	ax.legend()

def pairp( x, A, b, C, d, PL  ):

	if PL != None:
		##  Return exponential Power Law of degree PL:  A / x**PL
		ret = A / ( x ** PL )
	else:
		ret = A * np.exp(-b*x) - C * np.exp(-d*x)

	return ret

	
def bind_energy(xv, DD, l, m, n,  A, b, C, d, plot, PL):

	e_arr =  np.zeros( (8, len(xv)) )
	pp    =  np.array(     [ ]      )
	sbi   =  np.array( 	   [ ]      )

	for i in range(len(xv)):
		eigs, se   = get_bond_integrals( xv[i], DD, l, m, n )
		sbi        = np.append( sbi, se  )
		e_arr[:,i] = eigs

	if PL != 'EPP':
		pp   = pairp( xv, A, b, C, d , PL) 
	else:
		##  These parameters have minimum at the right r0 now
		Ae  = [  0.0040306, -0.0020265  ]
		r0e =  7.28
		me  = [10, 6] 
		pe  = [0, 0]
		pp  = epl( Ae, r0e, xv, me, pe )

	bind = sbi + pp
	b_min = np.argmin(bind)

	

	mass         =  12. * 1.67*10**(-27) 

	ryd_to_joule =  13.606 * 1.6 * 10**(-19)
	bohr_to_m    =  5.29177208*10**(-11)
	m_to_cm      =  100
	sol 	     =  2.98 * 10**(8)  ## Speed of light

	dxv 		 =  xv[1] - xv[0]
	k_c 		 =  (  bind[ (b_min + 1) % len(bind)]   -   2 * bind[b_min]   +   bind[b_min -1] ) / dxv**2

	omegasq      =  ( k_c / mass  ) * ryd_to_joule / ( bohr_to_m**2 ) 

	wavenumber   =  np.sqrt( omegasq ) / ( 2. * np.pi * sol * m_to_cm ) 
	wv_exp       =  2061 
	hz 		     =  2 * np.pi * 62.38681 * 10**(12)
 
	print('Minimum binding energy coord = %s \n Wavenumber at minimum = %s cm^-1,   exp = %s cm^-1  '%( xv[b_min], wavenumber, wv_exp) )

	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(111)

		yy = np.array([np.min(bind), np.max(pp)])

		xx = [ xv[b_min] for i in yy]


		ax.plot( xv, sbi,  label = 'E_band',     linestyle = '-')
		ax.plot( xv, pp,   label = 'E_pair',     linestyle = '-')
		ax.plot( xv, bind, label = 'E_bind',     linestyle = '-')
		ax.plot( xx, yy,   label = 'E_bind min', linestyle = '-')

		Ae  = [  0.0040306, -0.0020265  ]
		r0e =  5.6
		me  = [10, 6] 
		pe  = [0, 0]

		ep  = epl( Ae, r0e, xv, me, pe )
		epb = ep + sbi
		epb_min = np.argmin(epb)

		yye  = np.array([np.min(epb), np.max(pp)])
		xxe = [ xv[epb_min] for i in yye]

		ax.plot( xv,  ep,   label = 'EPL',          linestyle = '-')
		ax.plot( xv,  epb,  label = 'EPL_bind',     linestyle = '-')
		ax.plot( xxe, yye,  label = 'EPL bind_min', linestyle = '-')

		omsq_epb = (  epb[epb_min + 1]   -   2 * epb[epb_min]   +   epb[epb_min -1] ) / dxv**2

		omsq_epb = ( omsq_epb / mass  ) * ryd_to_joule / ( bohr_to_m**2 ) 

		wvnb_epb =   np.sqrt( omsq_epb ) / ( 2. * np.pi * sol * m_to_cm ) 
		wv_exp     =  2061 

		
		
 
		print('Minimum epb binding energy coord = %s \n EPB wavenumber at minimum = %s cm^-1,   exp = %s cm^-1  '%( xv[epb_min], wvnb_epb, wv_exp) )

		if DD == 'Full':
			ax.set_title( 'Binding energy for diatomic oxygen: Full' )
		elif DD:
			ax.set_title( 'Binding energy for diatomic oxygen: DD' )
		else:
			ax.set_title( 'Binding energy for diatomic oxygen: Analytic' )

		ax.set_xlabel( 'x (bohr)' )
		ax.set_ylabel( 'Energy (ryd)' )
		ax.legend()

	return bind

xv = np.linspace(1.2, 3., 400)
#bond_int_dep(xv, True)
#bond_int_dep(xv, False)

A = 30233.5; b = 4; C = 0.; d = 0.

PL = 4

if PL == None:
	A = 516.5 / np.exp( - 2.282788994 * b ) 

l = 1. ; m = 0.; n = 0

PL = 4

#bind_energy(xv, 'Full',  l, m, n,   A, b, C, d, True, PL)
bind_energy(xv, 'Full',  l, m, n,   A, b, C, d, True, 'EPP')
#bind_energy(xv, 'Full',  l, m, n,   A, b, C, d, True, None)
#bind_energy(xv,  True,   l, m, n,   A, b, C, d, True, PL)
#bind_energy(xv, False, A, b, C, d, True)
plt.show()

"""


Eigenvalues of this matrix in Mathematica

np.array( [  [ s,   s,   0,   0,   0,   -sp, -sp, -sp ], 
			 [ s,   s,   sp,  sp,  sp,  0,   0,   0   ], 
			 [ 0,   sp,  p,   0,   0,   pps, 0,   0   ],
			 [ 0,   sp,  0,   p,   0,   0,   ppp, 0   ], 
			 [ 0,   sp,  0,   0,   p,   0,   0,   ppp ], 
			 [ -sp, 0,   pps, 0,   0,   p,   0,   0   ], 
			 [ -sp, 0,   0,   ppp, 0,   0,   p,   0   ],
			 [ -sp, 0,   0,   0,   ppp, 0,   0,   p   ]  ]  )


{  { s,    ss,   0,    0,    0,    -sp,  -sp,  -sp }, 
				     { ss,    s,   sp,   sp,   sp,   0,    0,    0   }, 
				     { 0,    sp,   p,    0,    0,    pps,  0,    0   },
				     { 0,    sp,   0,    p,    0,    0,    ppp,  0   }, 
				     { 0,    sp,   0,    0,    p,    0,    0,    ppp }, 
				     { -sp,  0,    pps,  0,    0,    p,    0,    0   }, 
				     { -sp,  0,    0,    ppp,  0,    0,    p,    0   },
				     { -sp,  0,    0,    0,    ppp,  0,    0,    p   }  } 

{  p - ppp, p + ppp, 



 Root[3 p sp^2 + ppp sp^2 + 
        2 pps sp^2 + (p^2 + p ppp + p pps + ppp pps - 3 sp^2) #1 + (-2 p -
                                                           ppp - pps) #1^2 + #1^3 &, 1], 
 Root[3 p sp^2 + ppp sp^2 + 
    2 pps sp^2 + (p^2 + p ppp + p pps + ppp pps - 3 sp^2) #1 + (-2 p -
        ppp - pps) #1^2 + #1^3 &, 2], 
 Root[3 p sp^2 + ppp sp^2 + 
    2 pps sp^2 + (p^2 + p ppp + p pps + ppp pps - 3 sp^2) #1 + (-2 p -
        ppp - pps) #1^2 + #1^3 &, 3], 
 Root[-2 p^2 s + 2 p ppp s + 2 p pps s - 2 ppp pps s + 3 p sp^2 - 
    ppp sp^2 - 
    2 pps sp^2 + (p^2 - p ppp - p pps + ppp pps + 4 p s - 2 ppp s - 
       2 pps s - 3 sp^2) #1 + (-2 p + ppp + pps - 2 s) #1^2 + #1^3 &, 
  1], Root[-2 p^2 s + 2 p ppp s + 2 p pps s - 2 ppp pps s + 3 p sp^2 -
     ppp sp^2 - 
    2 pps sp^2 + (p^2 - p ppp - p pps + ppp pps + 4 p s - 2 ppp s - 
       2 pps s - 3 sp^2) #1 + (-2 p + ppp + pps - 2 s) #1^2 + #1^3 &, 
  2], Root[-2 p^2 s + 2 p ppp s + 2 p pps s - 2 ppp pps s + 3 p sp^2 -
     ppp sp^2 - 
    2 pps sp^2 + (p^2 - p ppp - p pps + ppp pps + 4 p s - 2 ppp s - 
       2 pps s - 3 sp^2) #1 + (-2 p + ppp + pps - 2 s) #1^2 + #1^3 &, 
  3]}

eigexp1
	  ( - p**2 * s   +   p * ppp * s   +   p * pps * s   -   ppp * pps * s   +   3 * p * sp**2   -   ppp * sp**2 - 
 				2 * pps * sp**2   -   p**2 * ss   +   p * ppp * ss   +   p * pps * ss - 
 				ppp * pps * ss   +   (p**2   -   p * ppp   -   p * pps   +   ppp * pps   +   2 * p * s   -   ppp * s - 
    			pps * s   -   3 * sp**2   +   2 * p * ss   -   ppp * ss   -   pps * ss  ) * ev   +   ( -2 * p  +  ppp  + 
    			pps  -  s -  ss ) * ev**2   +  ev**3)


eigexp2
	return (  - p**2 * s   -   p * ppp * s   -   p * pps * s   -   ppp * pps * s   +   3 * p * sp**2   +   ppp * sp**2   + 
    			2 * pps * sp**2   +   p**2 * ss   +   p * ppp * ss   +   p * pps * ss + 
 				ppp * pps * ss   +   (p**2   +   p * ppp   +   p * pps   +   ppp * pps   +   2 * p * s   +   ppp * s   + 
    			pps * s   -   3 * sp**2   -   2 * p * ss   -   ppp * ss   -   pps * ss ) * ev   +   (-2 * p - ppp *  - 
    			pps - s + ss ) * ev**2   +   ev**3 )


  """
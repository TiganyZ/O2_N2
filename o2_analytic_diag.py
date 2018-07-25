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

	return e

def get_O2_eigs_dd( s, p, ss, sp, pps, ppp ):
	H = np.array( [  [ s,    ss,   0,    0,    0,    -sp,  -sp,  -sp ], 
				     [ ss,    s,   sp,   sp,   sp,   0,    0,    0   ], 
				     [ 0,    sp,   p,    0,    0,    pps,  0,    0   ],
				     [ 0,    sp,   0,    p,    0,    0,    ppp,  0   ], 
				     [ 0,    sp,   0,    0,    p,    0,    0,    ppp ], 
				     [ -sp,  0,    pps,  0,    0,    p,    0,    0   ], 
				     [ -sp,  0,    0,    ppp,  0,    0,    p,    0   ],
				     [ -sp,  0,    0,    0,    ppp,  0,    0,    p   ]  ]  )



	e = np.linalg.eigh(H)[0]
	e[-1] = -e[-1]
	return e

def get_bond_integrals(x, DD):
	
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

	if DD:
		eigs = get_O2_eigs_dd(s, p, ss, sps, pps, ppp)

	else:
		eigs = get_O2_eigs(s, p, ss, sps, pps, ppp)
		
	print(eigs)
	return eigs



def bond_int_dep(xv, DD):

	e_arr = np.zeros( (8, len(xv)) )

	for i in range(len(xv)):
		eigs = get_bond_integrals( xv[i], DD )
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

def pairp( x, A, b, C, d  ):
	return A * np.exp(-b*x) - C * np.exp(-d*x)

	
def bind_energy(xv, DD, A, b, C, d, plot):

	e_arr =  np.zeros( (8, len(xv)) )
	pp    =  np.array(     [ ]      )

	for i in range(len(xv)):
		eigs = get_bond_integrals( xv[i], DD )
		pp 	 = np.append( pp, pairp( xv[i], A, b, C, d ) )
		e_arr[:,i] = eigs

	sbi  = np.sum(e_arr , axis = 0) 
	bind = sbi + pp

	print('Minimum binding energy coord',  xv[np.argmin(bind)] )

	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(111)

		yy = np.array([0, np.max(pp)])
		xx = [ xv[np.argmin(bind)] for i in yy]

		ax.plot(xv, sbi,  label = 'E_band', linestyle = '-')
		ax.plot(xv, pp,   label = 'E_pair', linestyle = '-')
		ax.plot(xv, bind, label = 'E_bind', linestyle = '-')
		ax.plot( xx, yy,  label = 'E_bind min', linestyle = '-')

		Ae  = [  0.0040306, -0.0020265  ]
		r0e =  5.6
		me  = [10, 6] 
		pe  = [0, 0]

		ep = epl( Ae, r0e, xv, me, pe )

		ax.plot( xv, ep,  label = 'EPL', linestyle = '-')

		if DD:
			ax.set_title( 'Binding energy for diatomic oxygen: DD' )
		else:
			ax.set_title( 'Binding energy for diatomic oxygen: Analytic' )

		ax.set_xlabel( 'x (bohr)' )
		ax.set_ylabel( 'Energy (ryd)' )
		ax.legend()

	return bind

xv = np.linspace(2, 4., 200)
#bond_int_dep(xv, True)
#bond_int_dep(xv, False)

A = 100.; b = 14; C = 0.; d = 0.

A = 700. / np.exp( - 2.282788994 * b ) 

bind_energy(xv, True, A, b, C, d, True)
#bind_energy(xv, False, A, b, C, d, True)
plt.show()

"""

[ 223.87516842 -226.17356842 -563.71348003  223.89085006  -12.33559773
 -563.71348003 -226.18937953    8.10293895]

[-563.71348003 -226.18937953 -226.17356842  -12.33559773    8.10293895
  223.87516842  223.89085006 -561.41506829]

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
import sympy as sp
import numpy as np
from sympy import diff



class Surface:
    def __init__(self,u,v,p:tuple):
        self.u=u
        self.v=v
        self.p=p
        
        self.p_u=[diff(x,self.u) for x in self.p]
        self.p_v=[diff(x,self.v) for x in self.p]
        
        self.p_uu=[diff(x,self.u) for x in self.p_u]
        self.p_uv=[diff(x,self.v) for x in self.p_u]
        self.p_vv=[diff(x,self.v) for x in self.p_v]
        
        E=np.dot(self.p_u,self.p_u)
        F=np.dot(self.p_u,self.p_v)
        G=np.dot(self.p_v,self.p_v)
        self.E=E
        self.F=F
        self.G=G
        
        e=np.cross(self.p_u,self.p_v)
        e_mod=sp.sqrt(np.dot(e,e))
        self.e=[(x/e_mod).simplify() for x in e]
        
        L=np.dot(self.p_uu,self.e)
        M=np.dot(self.p_uv,self.e)
        N=np.dot(self.p_vv,self.e)
        self.L=L
        self.M=M
        self.N=N
        self.K = ((L*N-M**2)/(E*G-F**2)).simplify()
        self.H = ((E*N+G*L-2*F*M)/(2*E*G-2*F**2)).simplify()
        self.K=self.K.simplify()
        self.H=self.H.simplify()
        k1,k2=sp.symbols("k1,k2")
        
        k1k2=sp.solve([
            k1*k2-self.K,
            (k1+k2)-2*self.H
        ],(k1,k2))
        self.k1=k1k2[0][0].simplify()
        self.k2=k1k2[0][1].simplify()
        
    def val(self,u,v):
        return tuple(x.subs(self.u,u).subs(self.v,v).evalf() for x in self.p)

    def e_val(self,u,v):
        return tuple(x.subs(self.u,u).subs(self.v,v).evalf() for x in self.e)
    
    def I(self,du,dv):
        return (self.E*du*du + 2*self.F*du*dv + self.G*dv*dv).simplify()
    
    def II(self,du,dv):
        return (self.L*du*du + 2*self.M*du*dv + self.N*dv*dv).simplify()
    
    def III(self,du,dv):
        return (-self.K*self.I(du,dv)+2*self.H*self.II(du,dv)).simplify()
    
    def principal_directions(self,u0,v0):
        k1,k2=self.k1,self.k2
        
        E,F,G,L,M,N=self.E,self.F,self.G,self.L,self.M,self.N
        k1,k2,E,F,G,L,M,N=[m.subs(self.u,u0).subs(self.v,v0) for m in [k1,k2,E,F,G,L,M,N]]
        
        a,b=sp.symbols("a,b")
        
        p_u=sp.Array(self.p_u).subs(self.u,u0).subs(self.v,v0)
        p_v=sp.Array(self.p_v).subs(self.u,u0).subs(self.v,v0)
        w1=sp.nonlinsolve([
            L*a + M*b -k1*(E*a + F*b),
            M*a + N*b -k1*(F*a + G*b),
            np.dot(a*p_u+b*p_v,a*p_u+b*p_v)-1
        ],(a,b))
        w2=sp.nonlinsolve([
            L*a + M*b -k2*(E*a + F*b),
            M*a + N*b -k2*(F*a + G*b),
            np.dot(a*p_u+b*p_v,a*p_u+b*p_v)-1
        ],(a,b))
        
        _w1=w1.args[0][0]*p_u+w1.args[0][1]*p_v
        _w2=w2.args[1][0]*p_u+w2.args[1][1]*p_v
        return _w1.simplify(),_w2.simplify()
    
    def find_umbilic_points(self):
        sols=sp.solve([
            self.k1-self.k2
        ],(self.u,self.v))
        return sols
    


class SurfaceCurve:
    def __init__(self, u,v,s, surface, curve:list):
        self.u=u
        self.v=v
        self.s=s
        if isinstance(surface,list):
            self.surface=Surface(u,v,surface)
        else:
            assert surface.u==u and surface.v==v
            self.surface=surface
        #p = p(u(s), v(s)) - a surface curve
        self.curve_param=curve
        self.curve=[x.subs(u,curve[0]).subs(v,curve[1]) for x in self.surface.p]
        
        self.d2pdiffs2=sp.Array([diff(diff(self.curve[i],s),s) for i in range(3)]).simplify()
    
    def normal_curvature_vec(self):
        u,v=self.u,self.v
        dudiffs=diff(self.curve[0],self.s)
        dvdiffs=diff(self.curve[1],self.s)
        L,M,N=self.surface.L,self.surface.M,self.surface.N
        L,M,N=[x.subs(u,self.curve_param[0]).subs(v,self.curve_param[1]).simplify() for x in [L,M,N]]
        k_n = (L*dudiffs**2+2*M*dudiffs*dvdiffs+N*dvdiffs**2).simplify()
        return (sp.Array(self.surface.e)*k_n).simplify()
    
    def geodesic_curvature_vec(self):
        s=self.s
        K_n=self.normal_curvature_vec()
        return (self.d2pdiffs2-K_n).simplify()
    
    def curvature_vectors(self):
        return self.normal_curvature_vec(),self.geodesic_curvature_vec()
    
    
    
class MethodOfOrthonormalFrames:
    def __init__(self,surface:Surface):
        self.surface=surface
        p_u=self.surface.p_u
        p_v=self.surface.p_v
        p_u_mod=sp.sqrt(np.dot(p_u,p_u))
        self.e1=[x/p_u_mod for x in p_u]
        _0 = np.dot(p_v,self.e1)
        _1 = [_0*x for x in self.e1]
        e2=[p_v[i]-_1[i] for i in range(len(p_v))]
        e2_mod=sp.sqrt(np.dot(e2,e2)).simplify()
        self.e2=[x/e2_mod for x in e2]
        self.e3=np.cross(self.e1,self.e2)
        self.du,self.dv=sp.symbols("du,dv")
    
    def calc_mat_A(self):
        a11,a12,a21,a22=sp.symbols("a11,a12,a21,a22")
        r1=sp.linsolve([
            a11*self.e1[0]+a12*self.e2[0]-self.surface.p_u[0],
            a11*self.e1[1]+a12*self.e2[1]-self.surface.p_u[1],
            a11*self.e1[2]+a12*self.e2[2]-self.surface.p_u[2]
        ],(a11,a12))
        r2=sp.linsolve([
            a21*self.e1[0]+a22*self.e2[0]-self.surface.p_v[0],
            a21*self.e1[1]+a22*self.e2[1]-self.surface.p_v[1],
            a21*self.e1[2]+a22*self.e2[2]-self.surface.p_v[2]
        ],(a21,a22))
        return sp.Matrix([
            [r1.args[0][0].simplify(),r1.args[0][1].simplify()],
            [r2.args[0][0].simplify(),r2.args[0][1].simplify()]
        ])
    
    def calc_mat_W(self):
        a11,a12,a13,a21,a22,a23,a31,a32,a33=sp.symbols("a11,a12,a13,a21,a22,a23,a31,a32,a33")
        b11,b12,b13,b21,b22,b23,b31,b32,b33=sp.symbols("b11,b12,b13,b21,b22,b23,b31,b32,b33")
        alpha=[
            [ a11, a12, a13],
            [ a21, a22, a23],
            [ a31, a32, a33]
        ]
        beta=[
            [ b11, b12, b13],
            [ b21, b22, b23],
            [ b31, b32, b33]
        ]
        eq=[self.mat_w_eq(i,alpha,beta) for i in range(9)]
        sols_alpha=sp.linsolve([x[0] for x in eq],(a11,a12,a13,a21,a22,a23,a31,a32,a33))
        sols_beta=sp.linsolve([x[1] for x in eq],(b11,b12,b13,b21,b22,b23,b31,b32,b33))
        a,b = sols_alpha.args[0],sols_beta.args[0]
        a=[x.simplify() for x in a]
        b=[x.simplify() for x in b]
        mat_a = [
            [a[0],a[1],a[2]],
            [a[3],a[4],a[5]],
            [a[6],a[7],a[8]]
        ]
        mat_b = [
            [b[0],b[1],b[2]],
            [b[3],b[4],b[5]],
            [b[6],b[7],b[8]]
        ]
        return sp.Matrix(mat_a),sp.Matrix(mat_b)
        
    def mat_w_eq(self,eq_num,alpha,beta):
        e1,e2,e3=self.e1,self.e2,self.e3
        """e1,e2,e3=[
            sp.symbols("x1,y1,z1"),
            sp.symbols("x2,y2,z2"),
            sp.symbols("x3,y3,z3")
        ]"""
        
        u,v=self.surface.u,self.surface.v
        vectors=[e1,e2,e3]
        vec_num=eq_num//3
        coord_num=eq_num%3
        
        vec=vectors[vec_num]
        dxdu=sp.diff(vec[coord_num],u)
        dxdv=sp.diff(vec[coord_num],v)

        eq1 = -dxdu-vectors[0][coord_num]*alpha[0][vec_num]\
            -vectors[1][coord_num]*alpha[1][vec_num]\
            -vectors[2][coord_num]*alpha[2][vec_num]
        
        eq2 = -dxdv-vectors[0][coord_num]*beta[0][vec_num]\
            -vectors[1][coord_num]*beta[1][vec_num]\
            -vectors[2][coord_num]*beta[2][vec_num]
        
        
        return eq1,eq2
    
    def calc_thetas(self):
        A = self.calc_mat_A()
        du,dv=self.du,self.dv
        return [A[0,0]*du+A[0,1]*dv,A[1,0]*du+A[1,1]*dv]
    
    def calc_mat_B(self):
        A = self.calc_mat_A()
        WA,WB = self.calc_mat_W()
        b11,b12,b21,b22=sp.symbols("b11,b12,b21,b22")
        sol=sp.linsolve([
            -WA[0,2]+A[0,0]*b11+A[1,0]*b12,
            -WA[1,2]+A[0,0]*b21+A[1,0]*b22,
            -WB[0,2]+A[0,1]*b11+A[1,1]*b12,
            -WB[1,2]+A[0,1]*b21+A[1,1]*b22
        ],(b11,b12,b21,b22))
        b=sol.args[0]
        return sp.Matrix([
            [b[0],b[1]],
            [b[2],b[3]]
        ])
    
    
    
class StructureEquations:
    def __init__(self,surface):
        self.surface=surface
        self.u=surface.u
        self.v=surface.v
        self.du,self.dv=sp.symbols("du,dv")
        self.ort=MethodOfOrthonormalFrames(self.surface)
        self.W_a,self.W_b = self.ort.calc_mat_W()
        self.A = self.ort.calc_mat_A()
        #A <=> the exterior product
        self.duAdv = sp.symbols("duAdv")
        
    def d_theta(self,num):
        """
        Using the 1st structure equation:
        d(theta_i) = theta_j A w_i_1 + theta_j A w_i_2 
        where j=(i+1)%2 and A denotes the exterior product
        """
        assert num in [1,2]
        i=num-1
        i0=num%2
        m=sp.Matrix([
            [self.A[i,0],self.A[i,1]],
            [self.W_a[i0,i],self.W_b[i0,i]]
        ])
        return m.det()*self.duAdv
    
    def d_w21(self):
        """
        Using the 2nd structure equation:
        (fundamental equation 1 of the theory of surfaces):
        d(w_2_1) = K * theta_1 A theta2
        where A denotes the exterior product
        """
        return self.surface.K*self.A.det()*self.duAdv
    
        
    
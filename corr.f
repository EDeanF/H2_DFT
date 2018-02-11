      subroutine eps(n,m,rs,epsc)
      implicit none
      integer n,m,i,j
      real*8 gama,beta1,beta2,A,B,C,D
      real*8 rs(n,m),epsc(n,m)
C     INITIALIZE PARAMETERS
      gama = -0.1423
      beta1 = 1.0529
      beta2 = 0.3334
      A = 0.0311
      B = -0.048
      C = 0.002
      D = -0.0116
Cf2py intent(in) n,m,rs
Cf2py intent(out) epsc
      DO 10 j = 1, m
      DO 20 i = 1, n
      IF (rs(i,j) >= 1) THEN 
      epsc(i,j) = gama/(1+beta1*sqrt(rs(i,j))+beta2*rs(i,j))
      ELSE 
      epsc(i,j)=A*log(rs(i,j))+B+rs(i,j)*C*log(rs(i,j))+D*rs(i,j)
      ENDIF
20    CONTINUE
10    CONTINUE
      END

      subroutine vc(n,m,rs,Vco)
      implicit none
      integer n,m,i,j
      real*8 gama,beta1,beta2,A,B,C,D
      real*8 rs(n,m),Vco(n,m)
C     INITIALIZE PARAMETERS
      gama = -0.1423
      beta1 = 1.0529
      beta2 = 0.3334
      A = 0.0311
      B = -0.048
      C = 0.002
      D = -0.0116
Cf2py intent(in) n,m,rs
Cf2py intent(out) Vco
      DO 10 j = 1, m
      DO 20 i = 1, n
      IF (rs(i,j) >= 1) THEN
      Vco(i,j) = gama*(1+7/6*beta1*sqrt(rs(i,j))+
     +4/3*beta2*rs(i,j))/
     +((1+beta1*sqrt(rs(i,j))+beta2*rs(i,j))**2)
      ELSE
      Vco(i,j) = A*log(rs(i,j))+B-A/3+2/3*rs(i,j)*C*log(rs(i,j))+
     +(2*D-C)/3*rs(i,j)
      ENDIF
20    CONTINUE
10    CONTINUE
      END

      subroutine arr(n,m,A,tot)
      implicit none
      integer n,m,i,j
      real*8 tot
      real*8 A(n,m)
      tot = 0.
Cf2py intent(in) n,m,A
Cf2py intent(out) tot
      DO 10 j = 1, m
      DO 20 i = 1, n
      tot = tot + A(i,j)
20    CONTINUE
10    CONTINUE
      END

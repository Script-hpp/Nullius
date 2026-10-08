import math
def phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def pin(lo,hi,mu,s): return max(0,phi((hi-mu)/s)-phi((lo-mu)/s))
U=[("kipf",0.8136,0.0051,10,0.815,"rep"),("shchur_gcn",0.8117,0.0133,30,0.815,"rep"),("dlinear",0.3809,0.0139,5,0.375,"rep"),
   ("nlinear",0.3739,0.0013,5,0.374,"rep"),("sgc",0.8081,0.0009,10,0.810,"rep"),("gat",0.8280,0.0050,10,0.830,"rep"),
   ("dgi",0.8054,0.0074,10,0.823,"teil"),("shchur_gat",0.8140,0.0115,30,0.818,"rep"),("sparsetsf",0.3607,0.0011,5,0.362,"rep"),("rlinear",0.3704,0.0008,5,0.366,"rep")]
for tau in (0.0,0.003,0.005,0.008):
  tot=0;new=0;row=[]
  for n,mu,sd,k,t,c in U:
    # LCC: splits shared -> only init part; use sd/sqrt(k) as conservative seed part
    s=math.sqrt(sd**2/k*2+tau**2)
    p=pin(t-.01,t+.01,mu,s) if c=="rep" else pin(t-.03,t-.01,mu,s)+pin(t+.01,t+.03,mu,s)
    m=p**3+3*p*p*(1-p)
    if n in("sparsetsf","rlinear"): new+=m
    else: tot+=m
    row.append(f"{n}:{p:.2f}/{m:.2f}")
  print(f"tau={tau}: dev {tot:.1f}/8 new {new:.1f}/2 | "+" ".join(row))

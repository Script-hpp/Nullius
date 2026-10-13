"""Author FITS kernel with the ee039f0f live settings (L90, cut3, lr1e-3, batch32, 10 epochs, patience3), seeds 0-2.

Usage: uv run python scripts/review/fits_live_setting_probe.py <out.json>. No provider calls."""
import sys
sys.argv=[sys.argv[0],'--out',sys.argv[1]]
sys.path.insert(0,'.')
import scripts.review.fits_independent_reference as ref
a=ref.ADAPTER
for old,new in [("length,horizon,batch=360,96,128","length,horizon,batch=90,96,32"),
                ("cut_freq=106)","cut_freq=3)"),("lr=.0005)","lr=.001)"),
                ("range(1 if smoke else 50)","range(1 if smoke else 10)"),("if bad>=20","if bad>=3"),
                ("group['lr']=.0005*","group['lr']=.001*")]:
    assert old in a, old
    a=a.replace(old,new)
ref.ADAPTER=a
ref.main()

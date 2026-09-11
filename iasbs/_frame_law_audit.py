import json, torch, numpy as np, itertools
import _bridge_audit as A
dev="cuda:0"
Cmat = torch.tensor([[0.7,-0.2],[0.1,0.8],[-0.4,0.3],[0.2,-0.5]],dtype=torch.float64,device=dev)

import sys; sys.path.insert(0,'/home/RESEARCH/iasbs/rasbs')
import rasbs_port as RP
U,_ = RP.basis_rotation(dev)
U = U.to(torch.float64)

def load(f,key=None,rot=False):
    c=torch.load(f,map_location='cpu',weights_only=False)
    s = c['samples'] if key is None else c['extra']['samples_refined'][key]
    s = s.to(dev).to(torch.float64)
    if rot:                      # R-ASBS stores X in its own basis: X_r = U X
        s = U.transpose(0,1) @ s
    return s

def obs(X):
    d={}
    d['E_trCX']=float((X*Cmat[None]).sum((-1,-2)).mean())
    m=X.mean(0); d['EX']=[[float(v) for v in r] for r in m]
    V=X.reshape(X.shape[0],-1)
    Cv=torch.cov(V.T)
    d['var_diag']=[float(v) for v in torch.diagonal(Cv)]
    d['cov']=[[float(v) for v in r] for r in Cv]
    return d,V

ref_s5 = load('ckpt/stiefel_frame_s5_b1_mcmc.pt')
ref_600 = load('ckpt/stiefel_frame600_b1_mcmc.pt')
o_ref,V_ref = obs(ref_s5)
o_ref6,V_ref6 = obs(ref_600)

s2 = A.median_bandwidth(V_ref, cap=4000)
MN, REPS = 4000, 8
gg=torch.Generator(device=dev); gg.manual_seed(99)
floor_m, floor_sd = A.split_half_floor(V_ref, MN, s2, 16, gg)
cross = np.mean([A.mmd2_unbiased(V_ref[torch.randperm(V_ref.shape[0],generator=gg,device=dev)[:MN]],
                                 V_ref6[torch.randperm(V_ref6.shape[0],generator=gg,device=dev)[:MN]], s2)
                 for _ in range(REPS)])

def mmd_vs_ref(V):
    vals=[A.mmd2_unbiased(V[torch.randperm(V.shape[0],generator=gg,device=dev)[:MN]],
                          V_ref[torch.randperm(V_ref.shape[0],generator=gg,device=dev)[:MN]], s2)
          for _ in range(REPS)]
    return float(np.mean(vals)), float(np.std(vals)/np.sqrt(REPS))

sets = {
 'IASBS_199':   [('ckpt/stiefel_frame_s5_b1_seed%d.pt'%i,None) for i in range(5)],
 'IASBS600_199':[('ckpt/stiefel_frame600_b1_seed%d.pt'%i,None) for i in range(5)],
 'RASBS_199':   [('ckpt/rasbs_frame_b1_seed%d.pt'%i,None) for i in range(5)],
 'RASBS_398':   [('ckpt/rasbs_frame_b1_seed%d.pt'%i,398) for i in range(5)],
 'RASBS_796':   [('ckpt/rasbs_frame_b1_seed%d.pt'%i,796) for i in range(5)],
}
res={'bandwidth_sigma2':s2,'mmd_n':MN,'reps':REPS,
     'ref':{'s5':o_ref,'f600':o_ref6},
     'floor_splithalf_mcmc_s5':{'mean':floor_m,'sd':floor_sd},
     'floor_cross_mcmc_s5_vs_600':float(cross),'methods':{}}
for name,fs in sets.items():
    per=[]
    for f,k in fs:
        X=load(f,k,rot=name.startswith('RASBS')); o,V=obs(X)
        m,se=mmd_vs_ref(V); o['MMD2']=m; o['MMD2_se']=se
        per.append(o)
    agg={}
    agg['E_trCX']=[float(np.mean([p['E_trCX'] for p in per])),float(np.std([p['E_trCX'] for p in per],ddof=1))]
    agg['MMD2']=[float(np.mean([p['MMD2'] for p in per])),float(np.std([p['MMD2'] for p in per],ddof=1))]
    EX=np.array([p['EX'] for p in per]); agg['EX_mean']=EX.mean(0).tolist(); agg['EX_sd']=EX.std(0,ddof=1).tolist()
    VD=np.array([p['var_diag'] for p in per]); agg['var_diag_mean']=VD.mean(0).tolist()
    CV=np.array([p['cov'] for p in per]).mean(0)
    agg['cov_frob_err']=float(np.linalg.norm(CV-np.array(o_ref['cov'])))
    agg['EX_max_abs_err']=float(np.abs(EX.mean(0)-np.array(o_ref['EX'])).max())
    res['methods'][name]={'per_seed':per,'agg':agg}
    print(name, json.dumps(agg['E_trCX']), 'MMD2', agg['MMD2'], 'EXerr', agg['EX_max_abs_err'], 'covF', agg['cov_frob_err'], flush=True)
json.dump(res, open('/home/RESEARCH/iasbs/json/results_frame_law_audit.json','w'), indent=1)
print('floor', floor_m, floor_sd, 'cross', cross)
print('ref trCX', o_ref['E_trCX'], o_ref6['E_trCX'])

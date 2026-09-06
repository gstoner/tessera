#!/usr/bin/env python3
"""Read-only commissioning evidence for the incoming native RDNA4 profiler host."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def collect(argv):
    executable = shutil.which(argv[0])
    if executable is None:
        return dict(command=argv, status='missing')
    try:
        result = subprocess.run([executable,*argv[1:]],capture_output=True,text=True,timeout=45)
        return dict(command=[executable,*argv[1:]],returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(command=argv,status='timeout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-gfx',default='gfx1201')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    probes={name:collect(command) for name,command in {
        'agents':['rocminfo'], 'hip':['hipconfig','--full'],
        'profiler':['rocprofv3','--version'], 'counters':['rocprofv3','--list-avail'],
        'systems':['rocprof-sys-run','--version'], 'systems_help':['rocprof-sys-run','--help'],
    }.items()}
    native=platform.system()=='Linux' and not re.search('microsoft|wsl',platform.release(),re.I)
    agents=probes['agents'].get('stdout','')
    targets=sorted(set(re.findall(r'\bgfx[0-9a-f]+\b',agents)))
    reasons=[]
    if not native:
        reasons.append('owning host must be native Linux')
    if args.expected_gfx not in targets:
        reasons.append('expected GPU target not visible')
    if not os.access('/dev/kfd',os.R_OK|os.W_OK):
        reasons.append('/dev/kfd not readable and writable by this user')
    for name in ('agents','hip','profiler','counters','systems'):
        if probes[name].get('returncode')!=0:
            reasons.append(name+' probe did not succeed')
    # Successful enumeration is not proof that counters produce valid samples.
    args.output.write_text(json.dumps(dict(kernel=platform.release(),os_release=Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else '',
        expected_gfx=args.expected_gfx,observed_targets=targets,probes=probes,blockers=reasons,
        status='blocked' if reasons else 'ready_for_measurement',hardware_counter_proof=False),indent=2)+'\n')
    print('blocked: '+ '; '.join(reasons) if reasons else 'ready for measurement; counter validity remains unproved')
    return bool(reasons)


if __name__=='__main__':
    raise SystemExit(main())

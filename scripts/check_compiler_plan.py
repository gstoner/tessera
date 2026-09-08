#!/usr/bin/env python3
"""Check compiler-plan routing and optional PR log/task co-maintenance.

Only structured task and log fields are contracts. Historical prose and random
ID mentions are deliberately not interpreted as current work or status.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path('docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md')
LOG = Path('docs/audit/compiler/INTEGRATED_COMPILER_LOG.md')
ARCHIVE = Path('docs/audit/compiler/archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md')
ID = r'[A-Z][A-Za-z0-9.-]*'
LINK = re.compile(r'\[([^\]\n]+)\]\(([^\s)]+)\)')


def slug(title: str) -> str:
    return re.sub(r'[^\w\- ]', '', re.sub(r'[`*_]', '', title).lower()).replace(' ', '-')


def anchors(text: str) -> set[str]:
    result: set[str] = set()
    counts: Counter[str] = Counter()
    fenced = False
    for line in text.splitlines():
        if line.startswith(('```', '~~~')):
            fenced = not fenced
        if fenced:
            continue
        match = re.match(r'^#{1,6} (.+)$', line)
        if match:
            stem = slug(match[1]); n = counts[stem]; counts[stem] += 1
            result.add(stem + (f'-{n}' if n else ''))
    return result


def records(text: str) -> dict[str, dict[str, str]]:
    result = {}
    cut = ''
    current = None
    for line in text.splitlines():
        if re.fullmatch(r'## F[0-5]', line):
            cut = line[3:]
        match = re.fullmatch(r'### ('+ID+r')', line)
        if match:
            current = match[1]
            if current in result:
                raise ValueError(f'duplicate task ID {current}')
            result[current] = {'Cut': cut}
        elif line.startswith('## '):
            current = None
        elif current:
            field = re.fullmatch(r'- (Owner|Gate|Depends on|Start|Latest): (.+)', line)
            if field:
                if field[1] in result[current]:
                    raise ValueError(f'duplicate {field[1]} for {current}')
                result[current][field[1]] = field[2]
    return result


def routes(text: str) -> dict[str, tuple[str, str]]:
    result = {}
    section = text.split('## Routing index\n', 1)[1]
    for line in section.splitlines():
        match = re.fullmatch(r'\| ('+ID+r') \| (.+) \| (owner|successor|archive) \|', line)
        if match:
            ident, dest, relation = match.groups()
            if ident in result:
                raise ValueError(f'duplicate routing ID {ident}')
            link = LINK.fullmatch(dest)
            if not link:
                raise ValueError(f'routing destination must be one link: {ident}')
            result[ident] = (link[2], relation)
        elif line.startswith('| ') and not line.startswith(('| ID |', '|---')):
            raise ValueError(f'malformed routing row: {line}')
    return result


def entries(text: str) -> dict[str, str]:
    result = {}
    matches = list(re.finditer(r'^### (.+)$', text, re.M))
    for i, match in enumerate(matches):
        title = match[1]
        if not re.match(r'^\d{4}-\d{2}-\d{2} — .+', title):
            raise ValueError(f'log heading must be date-first: {title}')
        anchor = slug(title)
        if anchor in result:
            raise ValueError(f'duplicate log anchor {anchor}')
        body = text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        if '<!-- entry-fields:end -->' not in body:
            raise ValueError(f'log {anchor} needs an entry-fields delimiter')
        body = body.split('<!-- entry-fields:end -->',1)[0]
        fields = {}
        for field in ('Owner', 'PRs', 'Outcome', 'Remaining', 'Evidence'):
            values = re.findall(r'^'+field+r': (.+)$', body, re.M)
            if len(values) != 1:
                raise ValueError(f'log {anchor} needs one {field}')
            fields[field] = values[0]
        owner = LINK.fullmatch(fields['Owner'])
        if owner is None or not re.fullmatch(ID, owner[1]):
            raise ValueError(f'log {anchor} needs a linked owner ID')
        result[anchor] = owner[1]
    return result


def check_transition(old_plan: str, old_log: str, new_plan: str, new_log: str) -> None:
    old_records, new_records = records(old_plan), records(new_plan)
    old_routes, new_routes = routes(old_plan), routes(new_plan)
    old_entries, new_entries = entries(old_log), entries(new_log)
    for anchor in new_entries.keys() - old_entries.keys():
        owner = new_entries[anchor]
        if old_records.get(owner) == new_records.get(owner) and old_routes.get(owner) == new_routes.get(owner):
            raise ValueError(f'new log entry {anchor} requires an update to {owner} or its disposition')


def link_source(path: Path, tracked: set[Path]) -> str:
    data = path.read_bytes()
    # Some test fixtures are deliberately non-UTF8. They are not navigation
    # sources unless they actually mention a document covered by this gate.
    if path.resolve() not in tracked and not any(p.name.encode() in data for p in tracked):
        return ''
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError(f'{path}: compiler navigation source must be UTF-8') from error


def check(root: Path = ROOT) -> None:
    plan, log = (root/PLAN).read_text(), (root/LOG).read_text()
    tasks, routing, history = records(plan), routes(plan), entries(log)
    if 'audit_role: reference' not in log.split('---',2)[1]:
        raise ValueError('engineering log must remain reference material')
    expected = ''
    for cut in ('F0','F2','F3','F4','F5'):
        candidates = [i for i,v in tasks.items() if v['Cut']==cut and v.get('Start')=='host-free']
        if candidates:
            expected += f'- {cut}: [{candidates[0]}](#{slug(candidates[0])}).\n'
    actual = plan.split('<!-- ready-view:start -->\n',1)[1].split('<!-- ready-view:end -->',1)[0]
    if actual != expected:
        raise ValueError('start view differs from the first host-free task in each cut')
    positions = {anchor:i for i,anchor in enumerate(history)}
    for ident, fields in tasks.items():
        if set(fields) != {'Cut','Owner','Gate','Depends on','Start','Latest'}:
            raise ValueError(f'incomplete task record {ident}')
        if fields['Start'] not in ('host-free','device','prerequisite'):
            raise ValueError(f'invalid Start value for {ident}')
        if routing.get(ident) != ('#'+slug(ident),'owner'):
            raise ValueError(f'active task {ident} needs its canonical routing row')
        for dep,_ in LINK.findall(fields['Depends on']):
            if dep not in routing or dep==ident:
                raise ValueError(f'unresolved/self dependency {ident} -> {dep}')
        latest = LINK.fullmatch(fields['Latest'])
        if not latest or latest[2].split('#',1)[-1] not in history:
            raise ValueError(f'{ident} Latest does not resolve to a log entry')
        linked = latest[2].split('#',1)[-1]
        owned = [positions[a] for a,o in history.items() if o==ident]
        if owned and positions[linked] < max(owned):
            raise ValueError(f'{ident} Latest predates its newest log entry')
    visiting, visited = set(), set()
    def visit(ident):
        if ident in visiting:
            raise ValueError(f'cyclic task dependency at {ident}')
        if ident in visited:
            return
        visiting.add(ident)
        for dep,_ in LINK.findall(tasks[ident]['Depends on']):
            if dep in tasks:
                visit(dep)
        visiting.remove(ident)
        visited.add(ident)
    for ident in tasks:
        visit(ident)
    for owner in history.values():
        if owner not in routing:
            raise ValueError(f'unresolved log owner {owner}')
    # Validate all new plan links, and inbound links to the three reorganized
    # documents. Do not reinterpret unrelated historical links as new contracts.
    tracked = {(root/p).resolve() for p in (PLAN,LOG,ARCHIVE)}
    candidates = list((root/'docs').rglob('*.md')) + list((root/'tests').rglob('*.md'))
    cache = {}
    for path in candidates:
        for _,url in LINK.findall(link_source(path,tracked)):
            if '://' in url or url.startswith('mailto:'):
                continue
            filepart,_,fragment = url.partition('#')
            target = (path.parent/filepart).resolve() if filepart else path.resolve()
            if path != root/PLAN and target not in tracked:
                continue
            if not target.is_file():
                raise ValueError(f'{path.relative_to(root)}: missing target {url}')
            if fragment:
                if target not in cache:
                    cache[target] = anchors(target.read_text())
                if fragment not in cache[target]:
                    raise ValueError(f'{path.relative_to(root)}: missing anchor {url}')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',help='PR base revision; compare new log entries to task/disposition updates')
    args=parser.parse_args()
    check()
    if args.base:
        base=subprocess.check_output(['git','merge-base',args.base,'HEAD'],cwd=ROOT,text=True).strip()
        def read(path):
            p=subprocess.run(['git','show',f'{base}:{path}'],cwd=ROOT,text=True,capture_output=True)
            return p.stdout if p.returncode==0 else None
        old_plan,old_log=read(PLAN),read(LOG)
        # First introduction is a migration: structural checks still apply.
        if old_plan is not None and old_log is not None:
            check_transition(old_plan,old_log,(ROOT/PLAN).read_text(),(ROOT/LOG).read_text())
    print('ok: compiler plan ownership, log links and start view')


if __name__=='__main__':
    main()

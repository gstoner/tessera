
import json, time, os

class Trace:
    def __init__(self):
        self.events = []
    def begin(self, name, cat="bench", args=None):
        ts = time.time() * 1e6
        self.events.append({"name":name,"cat":cat,"ph":"B","ts":ts,"pid":1,"tid":1,"args":args or {}})
        # NVTX optional
        try:
            import nvtx
            self._stack = getattr(self, "_stack", [])
            nvtx.range_push(name)
            self._stack.append(name)
        except Exception:
            pass
    def end(self):
        ts = time.time() * 1e6
        self.events.append({"ph":"E","ts":ts,"pid":1,"tid":1})
        try:
            import nvtx
            self._stack = getattr(self, "_stack", [])
            if self._stack:
                nvtx.range_pop()
                self._stack.pop()
        except Exception:
            pass
    def save(self, path):
        with open(path,"w") as f:
            json.dump({"traceEvents": self.events}, f)

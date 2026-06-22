import glob, zlib, re, sys

path = glob.glob("*.pdf")[0]
data = open(path, "rb").read()
print("filesize", len(data))

streams = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', data, re.DOTALL)
print("num streams", len(streams))

def tryd(b):
    # try as-is and trimmed, across window settings
    cands = [b, b.strip(b"\r\n"), b.lstrip(b"\r\n"), b.rstrip(b"\r\n")]
    for c in cands:
        for wbits in (15, -15, 47, 31):
            try:
                d = zlib.decompressobj(wbits)
                out = d.decompress(c) + d.flush()
                if out:
                    return out
            except Exception:
                pass
    # partial/streaming decompress to salvage what we can
    for wbits in (15, -15):
        try:
            d = zlib.decompressobj(wbits)
            out = b""
            for i in range(0, len(b), 4096):
                try:
                    out += d.decompress(b[i:i+4096])
                except Exception:
                    break
            if out:
                return out
        except Exception:
            pass
    return None

texts = []
ok = 0
for s in streams:
    out = tryd(s)
    if out:
        ok += 1
        texts.append(out)
print("decompressed ok:", ok, "/", len(streams))

def unescape(s):
    return (s.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\"))

alltext = []
for out in texts:
    try:
        txt = out.decode("latin-1")
    except Exception:
        continue
    # TJ arrays: extract parenthesized chunks
    for arr in re.findall(r'\[(.*?)\]\s*TJ', txt, re.DOTALL):
        parts = re.findall(r'\((?:[^()\\]|\\.)*\)', arr)
        line = "".join(unescape(p[1:-1]) for p in parts)
        if line.strip():
            alltext.append(line)
    # plain Tj
    for m in re.findall(r'\((?:[^()\\]|\\.)*\)\s*Tj', txt):
        s2 = re.findall(r'\((?:[^()\\]|\\.)*\)', m)
        if s2:
            line = unescape(s2[0][1:-1])
            if line.strip():
                alltext.append(line)

joined = "\n".join(alltext)
print("TEXT_LEN", len(joined))
open("pdf_extracted.txt", "w").write(joined)
print("---- preview ----")
print(joined[:3000])

import glob, zlib
import pikepdf

path = glob.glob("*.pdf")[0]
pdf = pikepdf.open(path)
print("pages", len(pdf.pages))

# inspect first few content streams raw bytes
count = 0
for objid in range(1, 60):
    try:
        obj = pdf.get_object((objid, 0))
    except Exception:
        continue
    try:
        import pikepdf
        if pikepdf.Stream and isinstance(obj, pikepdf.Stream):
            raw = obj.read_raw_bytes()
            head = raw[:4]
            print(objid, "rawlen", len(raw), "head", head.hex(), head)
            # zlib header should start 0x78
            try:
                dec = zlib.decompress(raw)
                print("   OK decompressed", len(dec))
            except Exception as e:
                print("   FAIL", e)
            count += 1
            if count > 8:
                break
    except Exception as e:
        pass

# Check for CRLF corruption signature: count of 0d0a vs 0a
data = open(path, "rb").read()
print("0d0a count", data.count(b"\r\n"), "lone 0a", data.count(b"\n"), "lone 0d", data.count(b"\r"))

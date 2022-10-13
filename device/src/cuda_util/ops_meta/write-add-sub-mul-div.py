from itertools import product

ty_list = [("f32", "float"), ("f64", "double")]
op_list = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]

for (tyname, ty), (opname, op) in product(ty_list, op_list):
    print(f"""
#define ULL unsigned long long
extern "C" __global__ void
{opname}_{tyname} (
    {ty}* x, {ty}* y, {ty}* z,
    ULL len
) {{
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {{
        z[i] = x[i] {op} y[i];
    }}
}}
#undef ULL
    """)

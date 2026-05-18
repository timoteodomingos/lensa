# import asyncio
# import re

# import aiohttp


# async def main():
#     async with aiohttp.ClientSession() as session:
#         headers = {
#             "Content-Type": "text/plain;charset=UTF-8",
#             "Accept": "text/x-component",
#             "Next-Action": "7f35a5c5c258d57a93766ef677eb48a6b3d88e20eb",
#         }
#         async with session.post(
#             f"https://virksomhet.brreg.no/en/oppslag/enheter/810034882",
#             headers=headers,
#             json=["810034882", "2024"],
#         ) as response:
#             print(response.status)
#             data = await response.read()

#             pdf_start = data.find(b"%PDF")
#             pdf_end = data.find(b"%%EOF") + len(b"%%EOF")
#             pdf_bytes = data[pdf_start:pdf_end]
#             with open("test.pdf", "wb") as f:
#                 f.write(pdf_bytes)


# if __name__ == "__main__":
#     asyncio.run(main())
#
import asyncio

import aiohttp


async def fetch_annual_report(org_nr: str, year: str):
    url = f"https://data.brreg.no/regnskapsregisteret/regnskap/aarsregnskap/kopi/{org_nr}/{year}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            print(resp.status)
            if resp.status == 200:
                pdf_bytes = await resp.read()
                filename = f"{org_nr}_{year}.pdf"
                with open(filename, "wb") as f:
                    f.write(pdf_bytes)
                print(f"Saved {len(pdf_bytes)} bytes to {filename}")
            else:
                print(await resp.text())


if __name__ == "__main__":
    asyncio.run(fetch_annual_report("810034882", "1900"))

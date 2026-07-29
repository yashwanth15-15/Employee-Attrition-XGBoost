import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            print("Navigating to docs...")
            await page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
            
            print("Clicking Authorize button...")
            await page.click("button.authorize")
            
            print("Filling credentials...")
            await page.fill(".auth-container input[type='text']", "admin")
            await page.fill(".auth-container input[type='password']", "Admin@123")
            
            print("Submitting...")
            await page.click(".auth-container button[type='submit']")
            
            await page.wait_for_timeout(2000)
            
            print("Closing modal...")
            await page.click(".auth-container button.btn-done")
            
            await page.wait_for_timeout(500)
            
            print("Expanding /api/v1/auth/me...")
            await page.click("#operations-Authentication-get_my_profile_api_v1_auth_me_get .opblock-summary-control")
            
            await page.wait_for_timeout(500)
            
            print("Clicking Try it out...")
            await page.click("#operations-Authentication-get_my_profile_api_v1_auth_me_get .try-out__btn")
            
            await page.wait_for_timeout(500)
            
            print("Executing...")
            await page.click("#operations-Authentication-get_my_profile_api_v1_auth_me_get .execute")
            
            await page.wait_for_timeout(1000)
            
            print("Extracting curl command...")
            curl = await page.inner_text("#operations-Authentication-get_my_profile_api_v1_auth_me_get .curl-command span")
            print("---- CURL ----")
            print(curl)
            print("--------------")
            
            status = await page.inner_text("#operations-Authentication-get_my_profile_api_v1_auth_me_get .responses-table .response-col_status")
            print("Status:", status)
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(run())

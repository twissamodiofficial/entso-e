# Forecast dashboard

This is a small read-only Next.js dashboard for the Supabase-backed forecast
pipeline. It shows the newest forecast against actual load, plus daily MAPE and
interval coverage.

## Local development

```sh
cp .env.example .env.local
npm install
npm run dev
```

`SUPABASE_SECRET_KEY` is server-only. Do not rename it to `NEXT_PUBLIC_*` and
do not use it in client-side code.

## Vercel

Create a Vercel project pointing at this repository and set its Root Directory
to `dashboard`. Add `SUPABASE_URL` and `SUPABASE_SECRET_KEY` to the Production
environment. Vercel can then build the app with `npm run build`.

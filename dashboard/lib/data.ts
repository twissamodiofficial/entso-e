import 'server-only'

import { createClient } from '@supabase/supabase-js'

export type ForecastPoint = {
  forecast_date: string
  valid_at: string
  point_forecast_mw: number
  q10_forecast_mw: number | null
  q50_forecast_mw: number | null
  q90_forecast_mw: number | null
  actual_load_mw: number | null
}

export type DailyMetric = {
  valid_date: string
  evaluated_rows: number
  point_mae_mw: number | null
  point_mape_percent: number | null
  interval_coverage: number | null
}

export type DashboardData = {
  modelVersion: string
  forecastDate: string
  issuedAt: string
  forecast: ForecastPoint[]
  dailyMetrics: DailyMetric[]
}

function serverClient() {
  const url = process.env.SUPABASE_URL
  const key =
    process.env.SUPABASE_SECRET_KEY ??
    process.env.SUPABASE_SERVICE_ROLE_KEY ??
    process.env.SUPABASE_KEY
  if (!url || !key) {
    throw new Error('SUPABASE_URL and a server-side Supabase key are required.')
  }
  return createClient(url, key, {
    auth: { autoRefreshToken: false, persistSession: false },
  })
}

export async function getDashboardData(): Promise<DashboardData> {
  const supabase = serverClient()
  const { data: runs, error: runError } = await supabase
    .from('forecast_runs')
    .select('run_id, model_version, forecast_date, issued_at')
    .order('issued_at', { ascending: false })
    .limit(365)

  if (runError) throw new Error(runError.message)
  const run = runs?.[0]
  if (!run) {
    return {
      modelVersion: '—',
      forecastDate: '—',
      issuedAt: '',
      forecast: [],
      dailyMetrics: [],
    }
  }

  const newestRunByDate = new Map<string, string>()
  for (const historicalRun of runs) {
    if (!newestRunByDate.has(historicalRun.forecast_date)) {
      newestRunByDate.set(historicalRun.forecast_date, historicalRun.run_id)
    }
  }
  const runIds = [...newestRunByDate.values()]

  const forecastQuery = runIds.length
    ? supabase
      .from('forecast_values')
      .select('run_id, valid_at, point_forecast_mw, q10_forecast_mw, q50_forecast_mw, q90_forecast_mw, actual_load_mw')
      .in('run_id', runIds)
      .order('valid_at', { ascending: true })
    : Promise.resolve({ data: [], error: null })
  const [{ data: forecast, error: forecastError }, { data: dailyMetrics, error: metricsError }] = await Promise.all([
    forecastQuery,
    supabase
      .from('forecast_daily_metrics')
      .select('valid_date, evaluated_rows, point_mae_mw, point_mape_percent, interval_coverage')
      .order('valid_date', { ascending: false })
      .limit(365),
  ])

  if (forecastError) throw new Error(forecastError.message)
  if (metricsError) throw new Error(metricsError.message)

  const dateByRun = new Map(runs.map((item) => [item.run_id, item.forecast_date]))
  return {
    modelVersion: run.model_version,
    forecastDate: run.forecast_date,
    issuedAt: run.issued_at,
    forecast: (forecast ?? []).map(({ run_id, ...item }) => ({
      ...item,
      forecast_date: dateByRun.get(run_id) ?? '',
    })) as ForecastPoint[],
    dailyMetrics: (dailyMetrics ?? []) as DailyMetric[],
  }
}

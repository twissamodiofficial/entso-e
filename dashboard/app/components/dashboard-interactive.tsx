'use client'

import { useMemo, useState } from 'react'
import type { DashboardData } from '../../lib/data'
import Charts from './charts'
import { ChartRange, dateBounds, dateInRange, localDate, todayInAmsterdam } from './range'

const percent = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 })

export default function DashboardInteractive({ data }: { data: DashboardData }) {
  const [range, setRange] = useState<ChartRange>('30')
  const today = todayInAmsterdam()
  const bounds = dateBounds(range, today)
  const filteredForecast = useMemo(
    () => data.forecast.filter((row) => dateInRange(row.forecast_date || localDate(row.valid_at), bounds)),
    [data.forecast, bounds],
  )
  const evaluated = filteredForecast.filter(
    (row) => row.actual_load_mw != null && row.actual_load_mw !== 0,
  )
  const rangeMape = evaluated.length
    ? evaluated.reduce(
      (total, row) => total + Math.abs((row.actual_load_mw! - row.point_forecast_mw) / row.actual_load_mw!) * 100,
      0,
    ) / evaluated.length
    : null
  const evaluatedDays = new Set(evaluated.map((row) => row.forecast_date)).size
  const latestMetric = data.dailyMetrics[0]
  const latestCoverage = latestMetric?.interval_coverage
  const actualCount = data.forecast.filter((row) => row.actual_load_mw !== null).length

  return <>
    <section className="cards" aria-label="Forecast summary">
      <article className="card"><span>Forecast date</span><strong>{data.forecastDate}</strong></article>
      <article className="card"><span>Model</span><strong className="small">{data.modelVersion}</strong></article>
      <article className="card"><span>MAPE for selected range</span><strong>{rangeMape == null ? '—' : `${percent.format(rangeMape)}%`}</strong><small className="card-note">{evaluatedDays} completed day{evaluatedDays === 1 ? '' : 's'} · {evaluated.length} rows</small></article>
      <article className="card"><span>Interval coverage</span><strong>{latestCoverage == null ? '—' : `${percent.format(latestCoverage * 100)}%`}</strong></article>
    </section>

    <section className="panel">
      <div className="panel-heading">
        <div><h2>Forecast vs actual</h2><p className="muted">Actual load appears after the forecast day closes.</p></div>
        <span className="chip">{actualCount}/{data.forecast.length || 0} actual rows</span>
      </div>
      <Charts
        forecast={data.forecast}
        dailyMetrics={data.dailyMetrics}
        range={range}
        onRangeChange={setRange}
      />
    </section>
  </>
}

import { getDashboardData } from '../lib/data'
import Charts from './components/charts'

export const dynamic = 'force-dynamic'

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const percent = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 })

export default async function Home() {
  const data = await getDashboardData()
  const latestMetric = data.dailyMetrics[0]
  const latestCoverage = latestMetric?.interval_coverage
  const actualCount = data.forecast.filter((row) => row.actual_load_mw !== null).length

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Europe / Amsterdam</p>
          <h1>Netherlands load forecast</h1>
          <p className="muted">Forecasts are issued for the local calendar day and reconciled when actuals arrive.</p>
        </div>
        <div className="status"><span className="dot" /> Live Supabase data</div>
      </header>

      <section className="cards" aria-label="Forecast summary">
        <article className="card"><span>Forecast date</span><strong>{data.forecastDate}</strong></article>
        <article className="card"><span>Model</span><strong className="small">{data.modelVersion}</strong></article>
        <article className="card"><span>Latest daily MAPE</span><strong>{latestMetric?.point_mape_percent == null ? '—' : `${percent.format(latestMetric.point_mape_percent)}%`}</strong></article>
        <article className="card"><span>Interval coverage</span><strong>{latestCoverage == null ? '—' : `${percent.format(latestCoverage * 100)}%`}</strong></article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div><h2>Forecast vs actual</h2><p className="muted">Actual load appears after the forecast day closes.</p></div>
          <span className="chip">{actualCount}/{data.forecast.length || 0} actual rows</span>
        </div>
        <Charts forecast={data.forecast} dailyMetrics={data.dailyMetrics} />
      </section>

      <footer>Updated from Supabase · Values shown in MW · {data.issuedAt ? `issued ${new Date(data.issuedAt).toLocaleString('en-GB', { timeZone: 'Europe/Amsterdam' })}` : 'no forecast run yet'}</footer>
    </main>
  )
}

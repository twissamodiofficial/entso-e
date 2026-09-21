import { getDashboardData } from '../lib/data'
import DashboardInteractive from './components/dashboard-interactive'

export const dynamic = 'force-dynamic'

export default async function Home() {
  const data = await getDashboardData()

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

      <DashboardInteractive data={data} />

      <footer>Updated from Supabase · Values shown in MW · {data.issuedAt ? `issued ${new Date(data.issuedAt).toLocaleString('en-GB', { timeZone: 'Europe/Amsterdam' })}` : 'no forecast run yet'}</footer>
    </main>
  )
}

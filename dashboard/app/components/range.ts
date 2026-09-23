export type ChartRange = 'today' | 'yesterday' | '7' | '30' | '90' | 'all'

export type DateBounds = { start: string; end: string } | null

export function localDate(value: string) {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/Amsterdam',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(value))
  return `${parts.find((part) => part.type === 'year')?.value}-${parts.find((part) => part.type === 'month')?.value}-${parts.find((part) => part.type === 'day')?.value}`
}

export function todayInAmsterdam() {
  return localDate(new Date().toISOString())
}

function subtractDays(date: string, days: number) {
  const value = new Date(`${date}T00:00:00Z`)
  value.setUTCDate(value.getUTCDate() - days)
  return value.toISOString().slice(0, 10)
}

export function dateBounds(range: ChartRange, today: string): DateBounds {
  if (range === 'all') return null
  if (range === 'yesterday') {
    const yesterday = subtractDays(today, 1)
    return { start: yesterday, end: yesterday }
  }
  const days = range === 'today' ? 1 : Number(range)
  return { start: subtractDays(today, days - 1), end: today }
}

export function dateInRange(date: string, bounds: DateBounds) {
  return !bounds || (date >= bounds.start && date <= bounds.end)
}

export function rangeLabel(range: ChartRange) {
  if (range === 'today') return 'Today'
  if (range === 'yesterday') return 'Yesterday'
  return range === 'all' ? 'All available dates' : `Last ${range} days`
}

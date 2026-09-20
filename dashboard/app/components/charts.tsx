'use client'

import { useEffect, useMemo, useState } from 'react'
import type { DailyMetric, ForecastPoint } from '../../lib/data'

const chartWidth = 960
const chartHeight = 350
const padding = { top: 24, right: 24, bottom: 46, left: 72 }

function scaleY(value: number, min: number, max: number) {
  const plotHeight = chartHeight - padding.top - padding.bottom
  return padding.top + (1 - (value - min) / Math.max(max - min, 1)) * plotHeight
}

function scaleX(index: number, length: number) {
  const plotWidth = chartWidth - padding.left - padding.right
  return padding.left + (index / Math.max(length - 1, 1)) * plotWidth
}

function linePath(values: Array<number | null>, min: number, max: number) {
  let output = ''
  values.forEach((value, index) => {
    if (value == null) return
    const command = values.slice(0, index).findLastIndex((item) => item != null) < 0 ? 'M' : 'L'
    output += `${command} ${scaleX(index, values.length).toFixed(1)} ${scaleY(value, min, max).toFixed(1)} `
  })
  return output.trim()
}

function formatMw(value: number) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/Amsterdam',
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function localDate(value: string) {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/Amsterdam',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(value))
  return `${parts.find((part) => part.type === 'year')?.value}-${parts.find((part) => part.type === 'month')?.value}-${parts.find((part) => part.type === 'day')?.value}`
}

function todayInAmsterdam() {
  return localDate(new Date().toISOString())
}

function subtractDays(date: string, days: number) {
  const value = new Date(`${date}T00:00:00Z`)
  value.setUTCDate(value.getUTCDate() - days)
  return value.toISOString().slice(0, 10)
}

function ChartAxis({ min, max }: { min: number; max: number }) {
  const ticks = Array.from({ length: 5 }, (_, index) => max - ((max - min) * index) / 4)
  return <>
    {ticks.map((value, index) => {
      const y = scaleY(value, min, max)
      return <g key={value}>
        <line className="gridline" x1={padding.left} x2={chartWidth - padding.right} y1={y} y2={y} />
        <text className="axis-label" x={padding.left - 10} y={y + 4} textAnchor="end">{formatMw(value)}</text>
        {index === 0 && <text className="axis-unit" x={padding.left - 10} y={y - 11} textAnchor="end">MW</text>}
      </g>
    })}
  </>
}

export default function Charts({ forecast, dailyMetrics }: { forecast: ForecastPoint[]; dailyMetrics: DailyMetric[] }) {
  const [range, setRange] = useState('30')
  const [hovered, setHovered] = useState<number | null>(null)
  const today = todayInAmsterdam()

  const selectedDates = useMemo(() => {
    if (range === 'all') return null
    const end = today
    const days = range === 'today' ? 1 : Number(range)
    return { start: subtractDays(end, days - 1), end }
  }, [range, today])

  const inRange = (date: string) => !selectedDates || (date >= selectedDates.start && date <= selectedDates.end)

  const filteredForecast = useMemo(
    () => forecast.filter((row) => inRange(row.forecast_date || localDate(row.valid_at))),
    [forecast, selectedDates],
  )

  const filteredMetrics = useMemo(
    () => [...dailyMetrics].filter((row) => inRange(row.valid_date)).reverse(),
    [dailyMetrics, selectedDates],
  )

  useEffect(() => setHovered(null), [range])

  if (!forecast.length && !dailyMetrics.length) return <div className="empty">No forecast metrics have been stored yet.</div>

  const values = filteredForecast.flatMap((row) => [row.point_forecast_mw, row.actual_load_mw ?? row.point_forecast_mw])
  const rawMin = values.length ? Math.min(...values) : 0
  const rawMax = values.length ? Math.max(...values) : 1
  const min = rawMin * 0.98
  const max = rawMax * 1.02
  const maxMape = Math.max(...filteredMetrics.map((row) => row.point_mape_percent ?? 0), 1)
  const hoveredRow = hovered == null ? null : filteredForecast[hovered]
  const rangeLabel = range === 'today' ? 'Today' : range === 'all' ? 'All available dates' : `Last ${range} days`

  return <div className="charts">
    <div className="range-heading">
      <div><strong>{rangeLabel}</strong><span> · Amsterdam local dates</span></div>
      <label>Range <select value={range} onChange={(event) => setRange(event.target.value)} aria-label="Chart date range">
        <option value="today">Today</option><option value="7">7 days</option><option value="30">30 days</option><option value="90">90 days</option><option value="all">All available</option>
      </select></label>
    </div>
    <div className="chart-block">
      <div className="legend"><span><i className="legend-line forecast-line" /> Forecast</span><span><i className="legend-line actual-line" /> Actual</span></div>
      <div className="line-chart-wrap">
        {filteredForecast.length ? <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Forecast and actual load chart">
            <ChartAxis min={min} max={max} />
            <path className="forecast-path" d={linePath(filteredForecast.map((row) => row.point_forecast_mw), min, max)} />
            <path className="actual-path" d={linePath(filteredForecast.map((row) => row.actual_load_mw), min, max)} />
            {filteredForecast.map((row, index) => <g key={`${row.forecast_date}-${row.valid_at}`}>
              <circle className="forecast-point" cx={scaleX(index, filteredForecast.length)} cy={scaleY(row.point_forecast_mw, min, max)} r="4"
                onMouseEnter={() => setHovered(index)} onMouseLeave={() => setHovered(null)} />
              {row.actual_load_mw != null && <circle className="actual-point" cx={scaleX(index, filteredForecast.length)} cy={scaleY(row.actual_load_mw, min, max)} r="4"
                onMouseEnter={() => setHovered(index)} onMouseLeave={() => setHovered(null)} />}
            </g>)}
          </svg> : <div className="chart-empty">No forecast rows for this date range.</div>}
        {hoveredRow && <div className="tooltip">
          <span>Forecast day: {hoveredRow.forecast_date}</span>
          <strong>{formatTime(hoveredRow.valid_at)}</strong>
          <span>Forecast: {formatMw(hoveredRow.point_forecast_mw)} MW</span>
          <span>Actual: {hoveredRow.actual_load_mw == null ? 'pending' : `${formatMw(hoveredRow.actual_load_mw)} MW`}</span>
          <span>Range: {hoveredRow.q10_forecast_mw == null || hoveredRow.q90_forecast_mw == null ? '—' : `${formatMw(hoveredRow.q10_forecast_mw)}–${formatMw(hoveredRow.q90_forecast_mw)} MW`}</span>
        </div>}
      </div>
      <div className="x-axis-labels"><span>{filteredForecast[0] ? formatTime(filteredForecast[0].valid_at) : ''}</span><span>{filteredForecast.at(-1) ? formatTime(filteredForecast.at(-1)!.valid_at) : ''}</span></div>
    </div>
    <div className="chart-block metric-chart">
      <div className="metric-heading"><div className="metric-title">Daily MAPE</div></div>
      <div className="mape-axis"><span>{maxMape.toFixed(1)}%</span><span>0%</span></div>
      {filteredMetrics.length ? <div className="bars">
          {filteredMetrics.map((row) => <div className="bar-wrap" key={row.valid_date} title={`${row.valid_date}: ${row.point_mape_percent?.toFixed(1) ?? '—'}%`}>
            <div className="bar" style={{ height: `${((row.point_mape_percent ?? 0) / maxMape) * 100}%` }} />
            <small>{row.valid_date.slice(5)}</small>
          </div>)}
        </div> : <div className="chart-empty">No daily MAPE for this date range.</div>}
    </div>
  </div>
}

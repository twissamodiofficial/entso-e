import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Netherlands Load Forecast',
  description: 'Live forecast versus actual load performance.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

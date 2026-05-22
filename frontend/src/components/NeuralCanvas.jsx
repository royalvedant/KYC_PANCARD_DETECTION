import { useEffect, useRef } from 'react'

const COLS = ['#00f5ff', '#bf00ff', '#ff0080', '#7c3aed', '#0ea5e9']
const N = 55

function makeNodes() {
  return Array.from({ length: N }, () => ({
    x: Math.random() * 1600,
    y: Math.random() * 900,
    vx: (Math.random() - 0.5) * 0.5,
    vy: (Math.random() - 0.5) * 0.5,
    r: 1.2 + Math.random() * 2,
    col: COLS[Math.floor(Math.random() * COLS.length)],
    ph: Math.random() * Math.PI * 2,
    sp: 0.018 + Math.random() * 0.022,
  }))
}

export default function NeuralCanvas() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const cx = canvas.getContext('2d')
    const nodes = makeNodes()
    let W, H, raf

    function resize() {
      W = canvas.width = window.innerWidth
      H = canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    function draw() {
      cx.clearRect(0, 0, W, H)
      const sx = W / 1600, sy = H / 900

      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const d = Math.sqrt(dx * dx + dy * dy)
          if (d < 160) {
            const a = (1 - d / 160) * 0.28
            const g = cx.createLinearGradient(
              nodes[i].x * sx, nodes[i].y * sy,
              nodes[j].x * sx, nodes[j].y * sy,
            )
            g.addColorStop(0, nodes[i].col)
            g.addColorStop(1, nodes[j].col)
            cx.beginPath()
            cx.moveTo(nodes[i].x * sx, nodes[i].y * sy)
            cx.lineTo(nodes[j].x * sx, nodes[j].y * sy)
            cx.strokeStyle = g
            cx.globalAlpha = a
            cx.lineWidth = 0.8
            cx.stroke()
          }
        }
      }

      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy; n.ph += n.sp
        if (n.x < 0 || n.x > 1600) n.vx *= -1
        if (n.y < 0 || n.y > 900) n.vy *= -1
        const p = 0.5 + 0.5 * Math.sin(n.ph)
        const grd = cx.createRadialGradient(n.x * sx, n.y * sy, 0, n.x * sx, n.y * sy, n.r * 4 * sx)
        grd.addColorStop(0, n.col)
        grd.addColorStop(1, 'transparent')
        cx.beginPath()
        cx.arc(n.x * sx, n.y * sy, n.r * 4 * sx, 0, Math.PI * 2)
        cx.fillStyle = grd; cx.globalAlpha = p * 0.45; cx.fill()
        cx.beginPath()
        cx.arc(n.x * sx, n.y * sy, n.r * p * sx, 0, Math.PI * 2)
        cx.fillStyle = n.col; cx.globalAlpha = 0.8; cx.fill()
      })

      cx.globalAlpha = 1
      raf = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas id="bgCanvas" ref={canvasRef} />
}

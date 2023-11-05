import React from 'react'

function App() {
  const months = {
    'Jan': 31,
    'Feb': 28,
    'Mar': 31,
    'Apr': 30,
    'May': 31,
    'Jun': 30,
    'Jul': 31,
    'Aug': 31,
    'Sep': 30,
    'Oct': 31,
    'Nov': 30,
    'Dec': 31
  }
  const [month, setMonth] = React.useState('Jan')
  const repeat = Array.from({ length: months[month] }, (_, i) => i)

  return (
    <div className='flex flex-col justify-center items-center h-screen'>
      <div className='mx-auto'>
        {
          Object.keys(months).map(m => (
            <button key={m} className={`${m === month ? 'bg-blue-900 text-white' : 'bg-white text-blue-900'}
              px-5 py-2 rounded-xl mx-auto border-2`} onClick={() => setMonth(m)}>{m}</button>
          ))
        }
      </div>
      <div className='grid grid-cols-7 place-content-start place-items-start justify-items-start w-fit'>
        {
          repeat.map(i => (
            <div key={i} className='self-start w-20 h-20 border-2 border-blue-900'>{i + 1}</div>
          ))
        }
      </div>
    </div>
  )
}

export default App

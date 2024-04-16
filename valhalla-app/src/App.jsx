import React from 'react'

import Map from './Map/Map'
import MainControl from './Controls'

class App extends React.Component {
  render() {
    return (
      <div>
        <Map />
        <MainControl />
      </div>
    )
  }
}

export default App

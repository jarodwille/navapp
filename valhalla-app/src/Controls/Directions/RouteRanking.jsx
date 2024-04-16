import React from 'react'
import PropTypes from 'prop-types'
import { Button, Dropdown } from 'semantic-ui-react'
import { connect } from 'react-redux'
import { submitRankings } from '../../actions/directionsActions'

class RouteRanking extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      routes: ['Route A', 'Route B', 'Standard'],
      comparisons: {
        'Route A vs Route B': '',
        'Route B vs Standard': '',
      },
      selectedRoutes: {
        first: 'Route A',
        second: 'Route B',
        third: 'Standard',
      },
    }
  }

  handleDropdownChange = (key, value) => {
    this.setState((prevState) => ({
      selectedRoutes: {
        ...prevState.selectedRoutes,
        [key]: value,
      },
    }))
  }

  handleComparisonChange = (comparison, value) => {
    this.setState((prevState) => ({
      comparisons: {
        ...prevState.comparisons,
        [comparison]: value,
      },
    }))
  }

  handleSubmit = () => {
    const { selectedRoutes } = this.state
    const { onSubmit } = this.props

    const rankings = {}

    // Extract selected routes
    const first = selectedRoutes.first
    const second = selectedRoutes.second
    const third = selectedRoutes.third

    // Get the values of the radio buttons for each comparison
    const comparison1Value = document.querySelector(
      'input[name="comparison1"]:checked'
    ).value
    const comparison2Value = document.querySelector(
      'input[name="comparison2"]:checked'
    ).value

    // Convert '>' and '=' to 1 and 2, respectively
    const comparison1Rank = comparison1Value === '>' ? 1 : 2
    const comparison2Rank = comparison2Value === '>' ? 1 : 2

    // Assign rankings based on the values of the radio buttons
    // first and second arguments
    rankings[`${first} vs ${second}`] = comparison1Rank
    const comparison1ReverseRank = comparison1Rank === 1 ? 3 : 2 // necessary since query is ordered
    rankings[`${second} vs ${first}`] = comparison1ReverseRank

    // second and third arguments
    rankings[`${second} vs ${third}`] = comparison2Rank
    const comparison2ReverseRank = comparison2Rank === 1 ? 3 : 2 // necessary since query is ordered
    rankings[`${third} vs ${second}`] = comparison2ReverseRank

    // first and third arguments
    if (comparison1Value === '=' && comparison2Value === '=') {
      rankings[`${first} vs ${third}`] = 2 // '='
      rankings[`${third} vs ${first}`] = 2 // '='
    } else {
      rankings[`${first} vs ${third}`] = 1 // '>'
      rankings[`${third} vs ${first}`] = 3 // '<'
    }

    onSubmit(rankings)
  }

  render() {
    const { routes, selectedRoutes } = this.state
    const dropdownOptions = routes.map((route) => ({
      key: route,
      text: route,
      value: route,
    }))

    const comparisonStyle = {
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      height: '60px', // Adjust the height as necessary to align with dropdowns
      marginLeft: '15px', // Increase spacing between dropdown and radio buttons
      marginRight: '15px',
    }

    const radioStyle = {
      marginBottom: '0px', // Decrease spacing between vertical radio buttons
    }

    return (
      <div>
        <p>Please compare routes A, B, with the standard route:</p>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Dropdown
            inline
            options={dropdownOptions}
            value={selectedRoutes.first}
            onChange={(e, { value }) =>
              this.handleDropdownChange('first', value)
            }
          />
          <div style={comparisonStyle}>
            <label style={radioStyle}>
              <input
                type="radio"
                name="comparison1"
                value=">"
                // Other radio button attributes like onChange, checked, etc.
              />{' '}
              {'>'}
            </label>
            <label>
              <input
                type="radio"
                name="comparison1"
                value="="
                // Other radio button attributes like onChange, checked, etc.
              />{' '}
              {'='}
            </label>
          </div>
          <Dropdown
            inline
            options={dropdownOptions}
            value={selectedRoutes.second}
            onChange={(e, { value }) =>
              this.handleDropdownChange('second', value)
            }
          />
          <div style={comparisonStyle}>
            <label style={radioStyle}>
              <input
                type="radio"
                name="comparison2"
                value=">"
                // Other radio button attributes like onChange, checked, etc.
              />{' '}
              {'>'}
            </label>
            <label>
              <input
                type="radio"
                name="comparison2"
                value="="
                // Other radio button attributes like onChange, checked, etc.
              />{' '}
              {'='}
            </label>
          </div>
          <Dropdown
            inline
            options={dropdownOptions}
            value={selectedRoutes.third}
            onChange={(e, { value }) =>
              this.handleDropdownChange('third', value)
            }
          />
        </div>
        <Button onClick={this.handleSubmit} style={{ marginTop: '15px' }}>
          Submit
        </Button>
      </div>
    )
  }
}

// export default RouteRanking

RouteRanking.propTypes = {
  onSubmit: PropTypes.func.isRequired,
}

const mapDispatchToProps = (dispatch) => ({
  onSubmit: (rankings) => dispatch(submitRankings(rankings)),
})

export default connect(null, mapDispatchToProps)(RouteRanking)

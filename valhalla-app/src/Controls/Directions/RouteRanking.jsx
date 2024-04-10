import React from 'react'
import PropTypes from 'prop-types'
import { Button, Dropdown } from 'semantic-ui-react'
import { connect } from 'react-redux'
import { submitRankings } from '../../actions/directionsActions'

class RouteRanking extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      rankings: {},
    }
  }

  handleChange = (comparison, rank) => {
    this.setState((prevState) => ({
      rankings: { ...prevState.rankings, [comparison]: rank },
    }))
  }

  handleSubmit = () => {
    const { rankings } = this.state
    // Add validation here if needed
    this.props.onSubmit(rankings)
  }

  render() {
    const { rankings } = this.state
    return (
      <div>
        <p>Please compare routes A, B, and C:</p>
        {[
          'Route A vs Standard',
          'Route B vs Standard',
          'Route A vs Route B',
        ].map((comparison) => {
          // Split the comparison string to dynamically generate options
          const [entity1, entity2] = comparison.split(' vs ')
          const options = [
            { key: '1', text: `${entity1} > ${entity2}`, value: '1' },
            { key: '2', text: `${entity1} = ${entity2}`, value: '2' },
            { key: '3', text: `${entity1} < ${entity2}`, value: '3' },
          ]

          return (
            <div key={comparison}>
              <label>{comparison}</label>
              <Dropdown
                placeholder="Enter Comparison"
                fluid
                selection
                options={options}
                value={rankings[comparison]}
                onChange={(e, { value }) =>
                  this.handleChange(comparison, value)
                }
              />
            </div>
          )
        })}
        <Button onClick={this.handleSubmit}>Submit</Button>
      </div>
    )
  }
}

RouteRanking.propTypes = {
  onSubmit: PropTypes.func.isRequired,
}

// Dispatch an action on submit:
const mapDispatchToProps = (dispatch) => ({
  onSubmit: (rankings) => dispatch(submitRankings(rankings)),
})

export default connect(null, mapDispatchToProps)(RouteRanking)

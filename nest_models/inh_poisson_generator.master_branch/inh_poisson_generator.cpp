/*
 *  inh_poisson_generator.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "inh_poisson_generator.h"

#include "event_delivery_manager_impl.h"
#include "kernel_manager.h"

#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"

// not sure
#include "exceptions.h"
//#include "network.h"
#include "integerdatum.h"
#include "arraydatum.h"
#include "numerics.h"
#include "universal_data_logger_impl.h"

#include <cmath>
#include <limits>

/* ----------------------------------------------------------------
 * Default constructors defining default parameter
 * ---------------------------------------------------------------- */

nest::inh_poisson_generator::Parameters_::Parameters_()
  : rate_times_(), // ms
    rate_values_() // spikes/ms
{
}

/* ----------------------------------------------------------------
 * Parameter extraction and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::inh_poisson_generator::Parameters_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::rate_times ] = DoubleVectorDatum( new std::vector< double_t >( rate_times_ ) );
  ( *d )[ names::rate_values ] = DoubleVectorDatum( new std::vector< double_t >( rate_values_ ) );

//  def< std::vector< double_t > >( d, names::rate_times, rate_times_ );
//  def< std::vector< double_t > >( d, names::rate_values, rate_values_ );
  //( *d )[ names::individual_spike_trains ] = individual_spike_trains_;
}

void
nest::inh_poisson_generator::Parameters_::set( const DictionaryDatum& d, Buffers_& b )
{
  const bool ut = updateValue< std::vector< double_t > >( d, names::rate_times, rate_times_ );
  const bool uv = updateValue< std::vector< double_t > >( d, names::rate_values, rate_values_ );

  if ( ut xor uv )
    throw BadProperty( "Rate times and values must be reset together." );

  if ( rate_times_.size() != rate_values_.size() )
    throw BadProperty( "Rate times and values have to be the same size." );

  // ensure amp times are strictly monotonically increasing
  if ( !rate_times_.empty() )
  {
    std::vector< double_t >::const_iterator prev = rate_times_.begin();
    for ( std::vector< double_t >::const_iterator next = prev + 1; next != rate_times_.end();
          ++next, ++prev )
      if ( *prev >= *next )
        throw BadProperty( "Rate times must strictly increasing." );
  }

  if ( ut && uv )
    b.idx_ = 0; // reset if we got new data
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::inh_poisson_generator::inh_poisson_generator()
  : Node(),
    device_(),
    P_()
  //, B_( *this )
{
}

nest::inh_poisson_generator::inh_poisson_generator( const inh_poisson_generator& n )
  : Node( n ),
    device_( n.device_ ),
    P_( n.P_ )
  //, B_( n.B_, *this )
{
}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */
void
nest::inh_poisson_generator::init_state_( const Node& proto )
{
  const inh_poisson_generator& pr = downcast< inh_poisson_generator >( proto );

  device_.init_state( pr.device_ );
}

void
nest::inh_poisson_generator::init_buffers_()
{
  device_.init_buffers();
  B_.idx_ = 0;
  B_.rate_ = 0;
  //B_.logger_.reset();
}

void
nest::inh_poisson_generator::calibrate()
{
  //B_.logger_.init(); // ensures initialization in case mm connected after Simulate *** (remove loger - no recordables)
  device_.calibrate();
  V_.h_ = Time::get_resolution().get_ms();
}

/* ----------------------------------------------------------------
 * Update function and event hook
 * ---------------------------------------------------------------- */

void
  // TODO changed
nest::inh_poisson_generator::update( Time const& origin, const long from, const long to )
{
  // TODO changed
//  assert( to >= 0 && ( delay ) from < Scheduler::get_min_delay() );
  assert(to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );

  assert( from < to );
  assert( P_.rate_times_.size() == P_.rate_values_.size() );

  // TODO changed
  const long t0 = origin.get_steps();

  // random number generator
  librandom::RngPtr rng = kernel().rng_manager.get_rng( get_thread() );


  // Skip any times in the past. Since we must send events proactively,
  // idx_ must point to times in the future.
  // TODO changed
  const long first = t0 + from;
  while ( B_.idx_ < P_.rate_times_.size()
    && Time( Time::ms( P_.rate_times_[ B_.idx_ ] ) ).get_steps() <= first )
    ++B_.idx_;

  // TODO changed
  for ( long offs = from; offs < to; ++offs )
  {
    const long curr_time = t0 + offs;

    // Keep the amplitude up-to-date at all times.
    // We need to change the amplitude one step ahead of time, see comment
    // on class SimulatingDevice.
      if ( B_.idx_ < P_.rate_times_.size()
        && curr_time + 1 == Time( Time::ms( P_.rate_times_[ B_.idx_ ] ) ).get_steps() )
      {
        B_.rate_ = P_.rate_values_[ B_.idx_ ] / 1000.0; // scale the rate to ms^-1
        B_.idx_++;
      }

        // store rate in Hz
        //B_.logger_.record_data( origin.get_steps() + offs);


      // create spikes
      if ( B_.rate_ > 0 && device_.is_active( Time::step( t0 + offs ) ) )
      {
          DSSpikeEvent se;
  // TODO changed
//          network()->send( *this, se, offs );
          kernel().event_delivery_manager.send( *this, se, offs );
      }
  }
}

void
nest::inh_poisson_generator::event_hook( DSSpikeEvent& e )
{
  // TODO changed
//  librandom::RngPtr rng = net_->get_rng( get_thread() );
  librandom::RngPtr rng = kernel().rng_manager.get_rng( get_thread() );
  V_.poisson_dev_.set_lambda( B_.rate_ * V_.h_ );
  long n_spikes = V_.poisson_dev_.ldev( rng );

  if ( n_spikes > 0 ) // we must not send events with multiplicity 0
  {
    e.set_multiplicity( n_spikes );
    e.get_receiver().handle( e );
  }
}

//void
//nest::inh_poisson_generator::handle( DataLoggingRequest& e )
//{
 // B_.logger_.handle( e );//
//}


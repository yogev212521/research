(define (domain logistics)
  (:requirements :strips :typing :negative-preconditions :equality)
  
  (:types
    location - location
    vehicle - object
    package - object
    truck airplane - vehicle
  )
  
  (:predicates
    (at ?obj - object ?loc - location)
    (in ?pkg - package ?veh - vehicle)
    (loaded ?pkg - package ?veh - vehicle)
  )
  
  (:action load
    :parameters (?pkg - package ?veh - vehicle ?loc - location)
    :precondition (and 
      (at ?pkg ?loc)
      (at ?veh ?loc)
      (not (in ?pkg ?veh))
    )
    :effect (and 
      (not (at ?pkg ?loc))
      (in ?pkg ?veh)
      (loaded ?pkg ?veh)
    )
  )
  
  (:action unload
    :parameters (?pkg - package ?veh - vehicle ?loc - location)
    :precondition (and 
      (in ?pkg ?veh)
      (at ?veh ?loc)
      (loaded ?pkg ?veh)
    )
    :effect (and 
      (at ?pkg ?loc)
      (not (in ?pkg ?veh))
      (not (loaded ?pkg ?veh))
    )
  )
  
  (:action drive
    :parameters (?truck - truck ?from - location ?to - location)
    :precondition (and 
      (at ?truck ?from)
      (not (= ?from ?to))
    )
    :effect (and 
      (not (at ?truck ?from))
      (at ?truck ?to)
    )
  )
  
  (:action fly
    :parameters (?plane - airplane ?from - location ?to - location)
    :precondition (and 
      (at ?plane ?from)
      (not (= ?from ?to))
    )
    :effect (and 
      (not (at ?plane ?from))
      (at ?plane ?to)
    )
  )
)

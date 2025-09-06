(define (problem logistics-problem-1)
  (:domain logistics)
  
  (:objects
    ; Locations
    city1 city2 city3 airport1 - location
    
    ; Vehicles
    truck1 truck2 - truck
    plane1 - airplane
    
    ; Packages
    package1 package2 package3 - package
  )
  
  (:init
    ; Initial locations of trucks
    (at truck1 city1)
    (at truck2 city2)
    
    ; Initial location of airplane
    (at plane1 airport1)
    
    ; Initial locations of packages
    (at package1 city1)
    (at package2 city2)
    (at package3 city3)
  )
  
  (:goal
    (and
      ; Goal: Get all packages to city3
      (at package1 city3)
      (at package2 city3)
      (at package3 city3)
    )
  )
)

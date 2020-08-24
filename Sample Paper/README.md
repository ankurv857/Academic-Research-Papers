## Evaluation Criteria ##
* Throughput using iperf
 * 2 connections on one subnet, one connection on other subnet
* Latency using ping
 * same as throughput using both subnets and multiple connections
* Accuracy 
* Detection Time (need to edit app to start timer when packet\_in received and end timer when alarm raised
* Coverage on all three cases (ensure spoofed packets cover all three cases and multiple IPs)

Notes: Baseline must run the malicious app but not the detection app and L2 learning must be edited to accept the custom event

## Limitations ##
* Fully passive adversary
 * simply argue the need for them to be active
* Only one switch is compromised
 * argue if they compromise all then there is no need for active
* Malicious app
 * argue the app simulates a compromised switch and does not ease detection
* Topology
 * not limitation but need to discuss our chosen topology
* Security App giving switch a false view of the network
* possibility of FN not being detected 
 * doubtful to congest link 
* We cannot handle very dynamic networks

## Diagrams ##
* Topology
* 3 cases of packet\_in
* how packet\_in and flow\_mod messages work
* SDN architecture
* Demonstration of an attack using packet\_in messages

## Things to Mention in the paper ##
* nMap scans are a previous way of doing network reconn
* NBI and SBI 
* an approach with infinte time and computational power could traceback all and where our tradeoff is
* Physical switch ports (event ports)
* In the intro discuss previous assumptions of trusted switches
* Corner case of a switch connected to external IP ranges reporting an internal IP from external network (possibly a 4th case)
* False positives theoretically arise if a switch begins to process packets too slowly 
* We didn't identify false postives but artificially introduced them to show it can happen
 * (see above for only case?)
* FALSE POSITIVE: compromised switch sends packets to neighboring switches and acts like it has never seen them making neighbors look suspicious.



## Presentation ##
* Title
 * myself Akash
* Motivation
* Background
 * packet\_in (animated)
  * breifly describe the 3 planes
 * Network Reconn in SDN (animated)
* Problem
* Design 
 * Approach 
  * We note identifying characteristic of spoofed switch port
  * 3 Cases (animated)
  * Solution using backtracking in each case
* Implementation
 * talk about file 
 * using own topo gathering etc
* Evaluation
 * simulate compromised switch
  * must change apps to listen on custom pi
 * experimental setup
  * hw
  * topo
   * only a portion of an enterprise network
  * traffic generator
  * performance metrics
   * throughput
   * latency 
   * detection time
   * accuracy
* Results
* Limitations
* Conclusion


// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title VPN Node Registry
 * @dev Manages registration and rewards for VPN nodes in the PhantomX network
 */
contract VPNNodeRegistry is AccessControl, Pausable, ReentrancyGuard {
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant NODE_ROLE = keccak256("NODE_ROLE");

    // VPN Node structure
    struct VPNNode {
        address owner;
        string endpoint;
        uint256 stake;
        uint256 totalBandwidth;
        uint256 rewardsClaimed;
        uint256 lastActive;
        bool isActive;
        uint256 reputation;
        Location location;
    }

    // Geographic location structure
    struct Location {
        string country;
        string region;
        string city;
        int256 latitude;
        int256 longitude;
    }

    // State variables
    IERC20 public phxToken;
    uint256 public constant MINIMUM_STAKE = 5000 * 10**18; // 5000 PHX
    uint256 public constant REWARD_RATE = 1 * 10**15; // 0.001 PHX per GB
    uint256 public constant REPUTATION_THRESHOLD = 80;
    uint256 public constant INACTIVE_THRESHOLD = 1 days;

    // Mappings
    mapping(address => VPNNode) public nodes;
    mapping(string => bool) public registeredEndpoints;
    address[] public activeNodes;

    // Events
    event NodeRegistered(address indexed owner, string endpoint);
    event NodeDeregistered(address indexed owner, string endpoint);
    event BandwidthReported(address indexed node, uint256 bandwidth);
    event RewardsClaimed(address indexed node, uint256 amount);
    event NodeStatusUpdated(address indexed node, bool isActive);
    event ReputationUpdated(address indexed node, uint256 newReputation);

    /**
     * @dev Constructor sets up roles and token contract
     * @param _phxToken Address of the PHX token contract
     */
    constructor(address _phxToken) {
        require(_phxToken != address(0), "Invalid token address");
        phxToken = IERC20(_phxToken);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
    }

    /**
     * @dev Registers a new VPN node
     * @param endpoint The node's endpoint address
     * @param location The node's geographic location
     */
    function registerNode(
        string memory endpoint,
        Location memory location
    ) public whenNotPaused nonReentrant {
        require(!registeredEndpoints[endpoint], "Endpoint already registered");
        require(nodes[msg.sender].stake == 0, "Node already registered");
        require(
            phxToken.balanceOf(msg.sender) >= MINIMUM_STAKE,
            "Insufficient PHX balance"
        );

        // Transfer stake to contract
        require(
            phxToken.transferFrom(msg.sender, address(this), MINIMUM_STAKE),
            "Stake transfer failed"
        );

        // Create new node
        nodes[msg.sender] = VPNNode({
            owner: msg.sender,
            endpoint: endpoint,
            stake: MINIMUM_STAKE,
            totalBandwidth: 0,
            rewardsClaimed: 0,
            lastActive: block.timestamp,
            isActive: true,
            reputation: 100,
            location: location
        });

        registeredEndpoints[endpoint] = true;
        activeNodes.push(msg.sender);
        _grantRole(NODE_ROLE, msg.sender);

        emit NodeRegistered(msg.sender, endpoint);
    }

    /**
     * @dev Deregisters a VPN node
     */
    function deregisterNode() public whenNotPaused nonReentrant {
        VPNNode storage node = nodes[msg.sender];
        require(node.stake > 0, "Node not registered");

        // Remove from active nodes
        _removeFromActiveNodes(msg.sender);

        // Return stake
        require(
            phxToken.transfer(msg.sender, node.stake),
            "Stake return failed"
        );

        registeredEndpoints[node.endpoint] = false;
        _revokeRole(NODE_ROLE, msg.sender);
        delete nodes[msg.sender];

        emit NodeDeregistered(msg.sender, node.endpoint);
    }

    /**
     * @dev Reports bandwidth usage for rewards
     * @param bandwidth Amount of bandwidth used in GB
     */
    function reportBandwidth(uint256 bandwidth) public whenNotPaused {
        require(hasRole(NODE_ROLE, msg.sender), "Not a registered node");
        VPNNode storage node = nodes[msg.sender];
        require(node.isActive, "Node not active");

        node.totalBandwidth += bandwidth;
        node.lastActive = block.timestamp;

        emit BandwidthReported(msg.sender, bandwidth);
    }

    /**
     * @dev Claims accumulated rewards
     */
    function claimRewards() public whenNotPaused nonReentrant {
        VPNNode storage node = nodes[msg.sender];
        require(node.stake > 0, "Node not registered");
        require(node.reputation >= REPUTATION_THRESHOLD, "Reputation too low");

        uint256 pendingRewards = calculateRewards(msg.sender);
        require(pendingRewards > 0, "No rewards to claim");

        node.rewardsClaimed += pendingRewards;
        require(
            phxToken.transfer(msg.sender, pendingRewards),
            "Reward transfer failed"
        );

        emit RewardsClaimed(msg.sender, pendingRewards);
    }

    /**
     * @dev Updates node reputation
     * @param node Node address
     * @param newReputation New reputation score
     */
    function updateReputation(
        address node,
        uint256 newReputation
    ) public onlyRole(OPERATOR_ROLE) {
        require(newReputation <= 100, "Invalid reputation score");
        require(nodes[node].stake > 0, "Node not registered");

        nodes[node].reputation = newReputation;
        emit ReputationUpdated(node, newReputation);

        // Deactivate node if reputation falls below threshold
        if (newReputation < REPUTATION_THRESHOLD && nodes[node].isActive) {
            nodes[node].isActive = false;
            _removeFromActiveNodes(node);
            emit NodeStatusUpdated(node, false);
        }
    }

    /**
     * @dev Calculates pending rewards for a node
     * @param node Node address
     */
    function calculateRewards(address node) public view returns (uint256) {
        VPNNode storage vpnNode = nodes[node];
        if (vpnNode.stake == 0 || vpnNode.reputation < REPUTATION_THRESHOLD) {
            return 0;
        }

        uint256 totalRewards = vpnNode.totalBandwidth * REWARD_RATE;
        return totalRewards - vpnNode.rewardsClaimed;
    }

    /**
     * @dev Returns all active nodes
     */
    function getActiveNodes() public view returns (address[] memory) {
        return activeNodes;
    }

    /**
     * @dev Returns node details
     * @param node Node address
     */
    function getNodeDetails(
        address node
    )
        public
        view
        returns (
            string memory endpoint,
            uint256 stake,
            uint256 bandwidth,
            uint256 reputation,
            bool isActive,
            Location memory location
        )
    {
        VPNNode storage vpnNode = nodes[node];
        return (
            vpnNode.endpoint,
            vpnNode.stake,
            vpnNode.totalBandwidth,
            vpnNode.reputation,
            vpnNode.isActive,
            vpnNode.location
        );
    }

    /**
     * @dev Removes a node from active nodes array
     */
    function _removeFromActiveNodes(address node) internal {
        for (uint256 i = 0; i < activeNodes.length; i++) {
            if (activeNodes[i] == node) {
                activeNodes[i] = activeNodes[activeNodes.length - 1];
                activeNodes.pop();
                break;
            }
        }
    }

    /**
     * @dev Pauses the contract
     */
    function pause() public onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }

    /**
     * @dev Unpauses the contract
     */
    function unpause() public onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
} 
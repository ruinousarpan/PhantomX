const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PhantomX Contracts", function () {
  let PhantomXToken;
  let VPNNodeRegistry;
  let token;
  let registry;
  let owner;
  let addr1;
  let addr2;
  let addrs;

  beforeEach(async function () {
    // Get signers
    [owner, addr1, addr2, ...addrs] = await ethers.getSigners();

    // Deploy PhantomX Token
    PhantomXToken = await ethers.getContractFactory("PhantomXToken");
    token = await PhantomXToken.deploy();
    await token.deployed();

    // Deploy VPN Node Registry
    VPNNodeRegistry = await ethers.getContractFactory("VPNNodeRegistry");
    registry = await VPNNodeRegistry.deploy(token.address);
    await registry.deployed();

    // Grant minter role to registry
    const minterRole = await token.MINTER_ROLE();
    await token.grantRole(minterRole, registry.address);

    // Mint initial tokens to addr1 for testing
    await token.mint(addr1.address, ethers.utils.parseEther("10000"));
  });

  describe("PhantomX Token", function () {
    it("Should set the right owner", async function () {
      expect(await token.hasRole(await token.DEFAULT_ADMIN_ROLE(), owner.address)).to.equal(true);
    });

    it("Should mint tokens to account", async function () {
      const amount = ethers.utils.parseEther("100");
      await token.mint(addr2.address, amount);
      expect(await token.balanceOf(addr2.address)).to.equal(amount);
    });

    it("Should allow staking", async function () {
      const stakeAmount = ethers.utils.parseEther("1000");
      await token.connect(addr1).approve(token.address, stakeAmount);
      await token.connect(addr1).stake(stakeAmount);

      const stake = await token.stakes(addr1.address);
      expect(stake.amount).to.equal(stakeAmount);
      expect(stake.isValidator).to.equal(false);
    });

    it("Should not allow staking below minimum", async function () {
      const smallAmount = ethers.utils.parseEther("100");
      await token.connect(addr1).approve(token.address, smallAmount);
      await expect(token.connect(addr1).stake(smallAmount)).to.be.revertedWith("Stake amount below minimum");
    });
  });

  describe("VPN Node Registry", function () {
    const endpoint = "vpn.phantomx.example.com";
    const location = {
      country: "US",
      region: "CA",
      city: "San Francisco",
      latitude: 37,
      longitude: -122
    };

    beforeEach(async function () {
      // Approve tokens for staking
      await token.connect(addr1).approve(registry.address, ethers.utils.parseEther("10000"));
    });

    it("Should register a new VPN node", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      const nodeDetails = await registry.getNodeDetails(addr1.address);
      
      expect(nodeDetails.endpoint).to.equal(endpoint);
      expect(nodeDetails.isActive).to.equal(true);
      expect(nodeDetails.reputation).to.equal(100);
    });

    it("Should not allow registering same endpoint twice", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      await token.connect(addr2).mint(addr2.address, ethers.utils.parseEther("10000"));
      await token.connect(addr2).approve(registry.address, ethers.utils.parseEther("10000"));
      
      await expect(
        registry.connect(addr2).registerNode(endpoint, location)
      ).to.be.revertedWith("Endpoint already registered");
    });

    it("Should allow reporting bandwidth", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      await registry.connect(addr1).reportBandwidth(100);
      
      const nodeDetails = await registry.getNodeDetails(addr1.address);
      expect(nodeDetails.bandwidth).to.equal(100);
    });

    it("Should calculate rewards correctly", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      await registry.connect(addr1).reportBandwidth(1000); // 1000 GB
      
      const rewards = await registry.calculateRewards(addr1.address);
      expect(rewards).to.equal(ethers.utils.parseEther("1")); // 1 PHX (0.001 PHX per GB * 1000)
    });

    it("Should update reputation", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      await registry.updateReputation(addr1.address, 90);
      
      const nodeDetails = await registry.getNodeDetails(addr1.address);
      expect(nodeDetails.reputation).to.equal(90);
    });

    it("Should deactivate node when reputation falls below threshold", async function () {
      await registry.connect(addr1).registerNode(endpoint, location);
      await registry.updateReputation(addr1.address, 70);
      
      const nodeDetails = await registry.getNodeDetails(addr1.address);
      expect(nodeDetails.isActive).to.equal(false);
    });
  });

  describe("Integration Tests", function () {
    it("Should handle full node lifecycle", async function () {
      // 1. Register node
      const endpoint = "test.vpn.phantomx.com";
      const location = {
        country: "US",
        region: "NY",
        city: "New York",
        latitude: 40,
        longitude: -74
      };

      await token.connect(addr1).approve(registry.address, ethers.utils.parseEther("10000"));
      await registry.connect(addr1).registerNode(endpoint, location);

      // 2. Report bandwidth
      await registry.connect(addr1).reportBandwidth(500);

      // 3. Earn rewards
      const initialBalance = await token.balanceOf(addr1.address);
      await registry.connect(addr1).claimRewards();
      const finalBalance = await token.balanceOf(addr1.address);
      expect(finalBalance).to.be.gt(initialBalance);

      // 4. Update reputation
      await registry.updateReputation(addr1.address, 95);
      let nodeDetails = await registry.getNodeDetails(addr1.address);
      expect(nodeDetails.reputation).to.equal(95);

      // 5. Deregister node
      await registry.connect(addr1).deregisterNode();
      nodeDetails = await registry.getNodeDetails(addr1.address);
      expect(nodeDetails.stake).to.equal(0);
    });
  });
}); 
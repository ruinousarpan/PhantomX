const hre = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy PhantomX Token
  console.log("Deploying PhantomX Token...");
  const PhantomXToken = await hre.ethers.getContractFactory("PhantomXToken");
  const token = await PhantomXToken.deploy();
  await token.deployed();
  console.log("PhantomX Token deployed to:", token.address);

  // Deploy VPN Node Registry
  console.log("Deploying VPN Node Registry...");
  const VPNNodeRegistry = await hre.ethers.getContractFactory("VPNNodeRegistry");
  const registry = await VPNNodeRegistry.deploy(token.address);
  await registry.deployed();
  console.log("VPN Node Registry deployed to:", registry.address);

  // Set up initial roles and permissions
  console.log("Setting up roles and permissions...");

  // Grant minter role to registry
  const minterRole = await token.MINTER_ROLE();
  await token.grantRole(minterRole, registry.address);
  console.log("Granted minter role to registry");

  // Mint initial supply to deployer
  const initialSupply = ethers.utils.parseEther("1000000"); // 1M PHX
  await token.mint(deployer.address, initialSupply);
  console.log("Minted initial supply to deployer");

  // Save deployment addresses
  const deploymentInfo = {
    network: hre.network.name,
    phantomxToken: token.address,
    vpnNodeRegistry: registry.address,
    deployer: deployer.address,
    timestamp: new Date().toISOString()
  };

  // Save to file
  const fs = require("fs");
  const deploymentPath = `deployments/${hre.network.name}.json`;
  fs.mkdirSync("deployments", { recursive: true });
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`Deployment info saved to ${deploymentPath}`);

  // Verify contracts on Etherscan if not on localhost
  if (hre.network.name !== "localhost" && hre.network.name !== "hardhat") {
    console.log("Waiting for block confirmations...");
    await token.deployTransaction.wait(5);
    await registry.deployTransaction.wait(5);

    console.log("Verifying contracts on Etherscan...");
    try {
      await hre.run("verify:verify", {
        address: token.address,
        constructorArguments: []
      });

      await hre.run("verify:verify", {
        address: registry.address,
        constructorArguments: [token.address]
      });
    } catch (error) {
      console.error("Error verifying contracts:", error);
    }
  }

  console.log("Deployment completed successfully!");
  return deploymentInfo;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 
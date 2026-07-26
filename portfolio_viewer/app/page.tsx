import portfolioData from "../public/data/portfolio.json";
import { PortfolioDashboard } from "./components/PortfolioDashboard";
import type { DashboardPayload } from "./portfolio-types";

export default function Home() {
  return (
    <PortfolioDashboard
      data={portfolioData as unknown as DashboardPayload}
    />
  );
}

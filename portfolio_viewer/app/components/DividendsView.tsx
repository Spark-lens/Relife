import { formatMoney } from "../portfolio-state.mjs";
import type { MarketPayload } from "../portfolio-types";

export function DividendsView({ market }: { market: MarketPayload }) {
  const total = market.dividends.reduce((sum, item) => sum + item.net, 0);
  const maximum = Math.max(
    ...market.dividendMonths.map((item) => Math.abs(item.net)),
    1,
  );

  return (
    <div className="dividend-layout">
      <section className="panel dividend-summary">
        <span className="eyebrow">INCOME</span>
        <p>累计净股息</p>
        <strong className={total >= 0 ? "positive" : "negative"}>
          {formatMoney(total, market.currency)}
        </strong>
        <small>{market.dividends.length} 笔股息记录</small>
      </section>
      <section className="panel dividend-chart-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">MONTHLY</span>
            <h2>月度净股息</h2>
          </div>
        </div>
        {market.dividendMonths.length === 0 ? (
          <div className="empty-state">暂无股息数据</div>
        ) : (
          <div className="dividend-bars" role="img" aria-label="月度净股息柱状图">
            {market.dividendMonths.map((item) => (
              <div className="bar-column" key={item.month}>
                <span>{formatMoney(item.net, market.currency)}</span>
                <div className="bar-track">
                  <i
                    className={item.net >= 0 ? "positive-bar" : "negative-bar"}
                    style={{
                      height: `${Math.max((Math.abs(item.net) / maximum) * 100, 4)}%`,
                    }}
                  />
                </div>
                <small>{item.month}</small>
              </div>
            ))}
          </div>
        )}
      </section>
      <section className="panel dividend-details">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">DETAILS</span>
            <h2>股息明细</h2>
          </div>
        </div>
        {market.dividends.length === 0 ? (
          <div className="empty-state">暂无股息记录</div>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>日期</th>
                  <th>标的</th>
                  <th>税前</th>
                  <th>税费调整</th>
                  <th>净额</th>
                </tr>
              </thead>
              <tbody>
                {market.dividends.map((item) => (
                  <tr key={`${item.date}:${item.symbol}`}>
                    <td className="muted-cell">{item.date}</td>
                    <td>
                      <div className="symbol-cell">
                        <strong>{item.symbol || "现金"}</strong>
                        <span>{item.name !== item.symbol ? item.name : ""}</span>
                      </div>
                    </td>
                    <td>{formatMoney(item.gross, market.currency)}</td>
                    <td
                      className={
                        (item.taxAdjustment ?? 0) < 0 ? "negative" : ""
                      }
                    >
                      {formatMoney(item.taxAdjustment, market.currency)}
                    </td>
                    <td className={item.net >= 0 ? "positive" : "negative"}>
                      {formatMoney(item.net, market.currency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

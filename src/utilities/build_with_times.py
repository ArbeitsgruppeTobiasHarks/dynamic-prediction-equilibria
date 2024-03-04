import sys
from datetime import datetime, timedelta, timezone
from time import time
from typing import Iterable, Optional, Tuple

from core.dynamic_flow import DynamicFlow
from core.flow_builder import FlowBuilder


def build_with_times(
    flow_builder: FlowBuilder,
    flow_id: Optional[int],
    reroute_interval: float,
    horizon: float,
    observe_commodities_indices: Optional[Iterable[int]] = None,
    suppress_log: bool = False,
) -> Tuple[DynamicFlow, float]:
    """
    @param flow_id: The id of the flow being built. Used only for logging.
    """
    generator = flow_builder.build_flow()
    start_time = last_milestone_time = time()
    flow = next(generator)
    start_date_time = (
        (datetime(1970, 1, 1) + timedelta(seconds=round(start_time)))
        .replace(tzinfo=timezone.utc)
        .astimezone(tz=None)
        .time()
    )

    last_log_time = float("NaN")
    if not suppress_log:
        print(
            f"Flow#{flow_id} built until phi={flow.phi}; Started At={start_date_time}"
        )
        last_log_time = time()
    milestone_interval = reroute_interval
    milestone = flow.phi

    def log_time():
        nonlocal now
        nonlocal last_log_time
        nonlocal start_time
        nonlocal remaining_time
        nonlocal new_milestone_time
        nonlocal finish_time
        elapsed = now - start_time
        log_msg = f"\rFlow#{flow_id} built until phi={flow.phi:.1f};"
        log_msg += f" Elapsed={timedelta(seconds=round(elapsed))};"
        log_msg += f" Remaining={timedelta(seconds=round(new_milestone_time + remaining_time - now))};"
        log_msg += f" Finished={finish_time};"
        if observe_commodities_indices is not None:
            log_msg += f" TravelTimes={[round(flow.avg_travel_time(i, flow.phi), 4) for i in observe_commodities_indices]};"
        # Clear rest of the line:
        log_msg += "\033[K"

        sys.stdout.write(log_msg)

        last_log_time = now

    while flow.phi < horizon:
        flow = next(generator)
        if flow.phi >= milestone:
            new_milestone_time = time()
            remaining_time = max(
                0,
                (
                    (horizon - flow.phi)
                    * (new_milestone_time - last_milestone_time)
                    / milestone_interval
                ),
            )
            finish_time = (
                (
                    datetime(1970, 1, 1)
                    + timedelta(seconds=round(new_milestone_time + remaining_time))
                )
                .replace(tzinfo=timezone.utc)
                .astimezone(tz=None)
                .time()
            )
            milestone += milestone_interval
            last_milestone_time = new_milestone_time
        now = time()
        if not suppress_log and now - last_log_time >= 1:
            log_time()

    if not suppress_log:
        log_time()

    print()
    elapsed = time() - start_time
    return flow, elapsed

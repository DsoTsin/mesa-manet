use crate::contract::Push;
use crate::error::{Error, ErrorKind, Result};
use std::collections::VecDeque;

pub trait QueueTransport {
    type Resource;
    type Point: Clone;
    fn validate_resource(&self, resource: &Self::Resource) -> Result<()>;
    fn submit(
        &mut self,
        pushes: &[Push],
        resources: &[Self::Resource],
        waits: &[Self::Point],
        signals: &[Self::Point],
    ) -> Result<Self::Point>;
    fn completed(&self, point: &Self::Point) -> Result<bool>;
    fn wait(&self, point: &Self::Point, timeout_ns: u64) -> Result<()>;
}

struct InFlight<R, P> {
    _resources: Vec<R>,
    _waits: Vec<P>,
    _signals: Vec<P>,
    point: P,
}

pub struct Queue<T: QueueTransport> {
    transport: T,
    pushes: Vec<Push>,
    resources: Vec<T::Resource>,
    waits: Vec<T::Point>,
    signals: Vec<T::Point>,
    in_flight: VecDeque<InFlight<T::Resource, T::Point>>,
    lost: bool,
    poisoned: bool,
}

impl<T: QueueTransport> Queue<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            pushes: Vec::new(),
            resources: Vec::new(),
            waits: Vec::new(),
            signals: Vec::new(),
            in_flight: VecDeque::new(),
            lost: false,
            poisoned: false,
        }
    }

    fn ready(&self) -> Result<()> {
        if self.lost {
            return Err(Error::new(ErrorKind::DeviceLost, "queue"));
        }
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::System,
                "queue requires teardown",
            ));
        }
        Ok(())
    }

    pub fn push(&mut self, push: Push) -> Result<()> {
        self.ready()?;
        self.pushes.push(push.validate()?);
        Ok(())
    }

    pub fn retain(&mut self, resource: T::Resource) -> Result<()> {
        self.ready()?;
        self.transport.validate_resource(&resource)?;
        self.resources.push(resource);
        Ok(())
    }

    pub fn wait_on(&mut self, point: T::Point) -> Result<()> {
        self.ready()?;
        self.waits.push(point);
        Ok(())
    }

    pub fn signal(&mut self, point: T::Point) -> Result<()> {
        self.ready()?;
        self.signals.push(point);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<T::Point> {
        self.ready()?;
        if self.pushes.last().is_some_and(|p| p.incomplete) {
            return Err(Error::new(
                ErrorKind::Invalid,
                "incomplete push batch",
            ));
        }
        match self.transport.submit(
            &self.pushes,
            &self.resources,
            &self.waits,
            &self.signals,
        ) {
            Ok(point) => {
                self.pushes.clear();
                self.in_flight.push_back(InFlight {
                    _resources: std::mem::take(&mut self.resources),
                    _waits: std::mem::take(&mut self.waits),
                    _signals: std::mem::take(&mut self.signals),
                    point: point.clone(),
                });
                Ok(point)
            }
            Err(error) => {
                self.lost = error.kind == ErrorKind::DeviceLost;
                self.poisoned = !self.lost;
                Err(error)
            }
        }
    }

    pub fn collect(&mut self) -> Result<usize> {
        self.ready()?;
        let mut count = 0;
        while let Some(first) = self.in_flight.front() {
            match self.transport.completed(&first.point) {
                Ok(true) => {
                    self.in_flight.pop_front();
                    count += 1;
                }
                Ok(false) => break,
                Err(error) => {
                    self.lost = error.kind == ErrorKind::DeviceLost;
                    return Err(error);
                }
            }
        }
        Ok(count)
    }

    pub fn wait_idle(&mut self, timeout_ns: u64) -> Result<()> {
        self.ready()?;
        if let Some(last) = self.in_flight.back() {
            self.transport.wait(&last.point, timeout_ns)?;
        }
        self.collect()?;
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.in_flight.len()
    }
    pub fn transport(&self) -> &T {
        &self.transport
    }
}

impl<T: QueueTransport> Drop for Queue<T> {
    fn drop(&mut self) {
        if self.poisoned && !self.lost {
            std::mem::forget(std::mem::take(&mut self.resources));
            std::mem::forget(std::mem::take(&mut self.waits));
            std::mem::forget(std::mem::take(&mut self.signals));
        }
        if let Some(last) = self.in_flight.back() {
            if self.transport.wait(&last.point, 5_000_000_000).is_err()
                && !self.lost
            {
                for flight in self.in_flight.drain(..) {
                    std::mem::forget(flight);
                }
            }
        }
    }
}

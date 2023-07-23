use std::marker::PhantomData;

use mlir_sys::{
    mlirModuleDestroy, mlirPDLPatternGet, mlirPatternSetAddOwnedPDLPattern,
    mlirRewritePatternSetGet, MlirPDLPatternModule, MlirRewritePatternSet,
};

use crate::{ir::Module, Context};

#[derive(Debug)]
pub struct RewritePatternSet<'c> {
    raw: MlirRewritePatternSet,
    _context: PhantomData<&'c Context>,
}

impl<'c> RewritePatternSet<'c> {
    pub fn new(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirRewritePatternSetGet(context.to_raw())) }
    }

    /// Adds a PDL pattern to the set.
    pub fn add_pdl_pattern(&mut self, pattern: PDLPatternModule<'c>) {
        unsafe {
            mlirPatternSetAddOwnedPDLPattern(self.raw, pattern.to_raw());
        }
    }

    /// Creates a rewrite pattern set from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirRewritePatternSet) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts a rewrite pattern set into a raw object.
    pub const fn to_raw(self) -> MlirRewritePatternSet {
        self.raw
    }
}

#[derive(Debug)]
pub struct PDLPatternModule<'c> {
    raw: MlirPDLPatternModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> PDLPatternModule<'c> {
    pub fn new(module: Module<'c>) -> Self {
        let pdl_module = unsafe { Self::from_raw(mlirPDLPatternGet(module.to_raw())) };
        std::mem::forget(module); // Make sure we don't destroy the module yet
        pdl_module
    }

    /// Creates a PDL pattern from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirPDLPatternModule) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts a PDL pattern into a raw object.
    pub const fn to_raw(self) -> MlirPDLPatternModule {
        self.raw
    }
}

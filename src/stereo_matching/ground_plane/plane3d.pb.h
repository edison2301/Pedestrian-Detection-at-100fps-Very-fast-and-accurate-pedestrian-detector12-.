// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: plane3d.proto

#ifndef PROTOBUF_plane3d_2eproto__INCLUDED
#define PROTOBUF_plane3d_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2004001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
// @@protoc_insertion_point(includes)

namespace doppia_protobuf {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_plane3d_2eproto();
void protobuf_AssignDesc_plane3d_2eproto();
void protobuf_ShutdownFile_plane3d_2eproto();

class Plane3d;

// ===================================================================

class Plane3d : public ::google::protobuf::Message {
 public:
  Plane3d();
  virtual ~Plane3d();
  
  Plane3d(const Plane3d& from);
  
  inline Plane3d& operator=(const Plane3d& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const Plane3d& default_instance();
  
  void Swap(Plane3d* other);
  
  // implements Message ----------------------------------------------
  
  Plane3d* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Plane3d& from);
  void MergeFrom(const Plane3d& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // required float offset = 1;
  inline bool has_offset() const;
  inline void clear_offset();
  static const int kOffsetFieldNumber = 1;
  inline float offset() const;
  inline void set_offset(float value);
  
  // required float normal_x = 2;
  inline bool has_normal_x() const;
  inline void clear_normal_x();
  static const int kNormalXFieldNumber = 2;
  inline float normal_x() const;
  inline void set_normal_x(float value);
  
  // required float normal_y = 3;
  inline bool has_normal_y() const;
  inline void clear_normal_y();
  static const int kNormalYFieldNumber = 3;
  inline float normal_y() const;
  inline void set_normal_y(float value);
  
  // required float normal_z = 4;
  inline bool has_normal_z() const;
  inline void clear_normal_z();
  static const int kNormalZFieldNumber = 4;
  inline float normal_z() const;
  inline void set_normal_z(float value);
  
  // @@protoc_insertion_point(class_scope:doppia_protobuf.Plane3d)
 private:
  inline void set_has_offset();
  inline void clear_has_offset();
  inline void set_has_normal_x();
  inline void clear_has_normal_x();
  inline void set_has_normal_y();
  inline void clear_has_normal_y();
  inline void set_has_normal_z();
  inline void clear_has_normal_z();
  
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  
  float offset_;
  float normal_x_;
  float normal_y_;
  float normal_z_;
  
  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(4 + 31) / 32];
  
  friend void  protobuf_AddDesc_plane3d_2eproto();
  friend void protobuf_AssignDesc_plane3d_2eproto();
  friend void protobuf_ShutdownFile_plane3d_2eproto();
  
  void InitAsDefaultInstance();
  static Plane3d* default_instance_;
};
// ===================================================================


// ===================================================================

// Plane3d

// required float offset = 1;
inline bool Plane3d::has_offset() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Plane3d::set_has_offset() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Plane3d::clear_has_offset() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Plane3d::clear_offset() {
  offset_ = 0;
  clear_has_offset();
}
inline float Plane3d::offset() const {
  return offset_;
}
inline void Plane3d::set_offset(float value) {
  set_has_offset();
  offset_ = value;
}

// required float normal_x = 2;
inline bool Plane3d::has_normal_x() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Plane3d::set_has_normal_x() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Plane3d::clear_has_normal_x() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Plane3d::clear_normal_x() {
  normal_x_ = 0;
  clear_has_normal_x();
}
inline float Plane3d::normal_x() const {
  return normal_x_;
}
inline void Plane3d::set_normal_x(float value) {
  set_has_normal_x();
  normal_x_ = value;
}

// required float normal_y = 3;
inline bool Plane3d::has_normal_y() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Plane3d::set_has_normal_y() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Plane3d::clear_has_normal_y() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Plane3d::clear_normal_y() {
  normal_y_ = 0;
  clear_has_normal_y();
}
inline float Plane3d::normal_y() const {
  return normal_y_;
}
inline void Plane3d::set_normal_y(float value) {
  set_has_normal_y();
  normal_y_ = value;
}

// required float normal_z = 4;
inline bool Plane3d::has_normal_z() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Plane3d::set_has_normal_z() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Plane3d::clear_has_normal_z() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Plane3d::clear_normal_z() {
  normal_z_ = 0;
  clear_has_normal_z();
}
inline float Plane3d::normal_z() const {
  return normal_z_;
}
inline void Plane3d::set_normal_z(float value) {
  set_has_normal_z();
  normal_z_ = value;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace doppia_protobuf

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_plane3d_2eproto__INCLUDED
